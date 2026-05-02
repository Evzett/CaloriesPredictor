import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from dataset import MultiModalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True



def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultiModalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
       
        self.image_model = timm.create_model(config.IMAGE_MODEL_NAME,
        pretrained=True, num_classes=0)

        self.ingr_embedding = nn.Embedding(config.NUM_INGREDIENTS, 128, padding_idx=0)

        self.ingr_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        self.ingr_proj = nn.Linear(128, config.HIDDEN_DIM)
        self.image_proj = nn.Linear(self.image_model.num_features, config.HIDDEN_DIM)
        self.num_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, config.HIDDEN_DIM)
        )

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 6, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )     

    def forward(self, ingredients_ids, ingredients_mask, image, mass, amount_ing):

        emb = self.ingr_embedding(ingredients_ids)  # (B, L, 128)

        attn_mask = (ingredients_mask == 0)

        attn_out, _ = self.ingr_attention(
            emb, emb, emb,
            key_padding_mask=attn_mask
        )

        mask = ingredients_mask.unsqueeze(-1)
        attn_out = attn_out * mask

        summed = attn_out.sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)

        ingr_features = summed / count
        ingr_emb = self.ingr_proj(ingr_features)

  
        image_features = self.image_model(image)
        image_emb = self.image_proj(image_features)

        num = torch.stack([mass, amount_ing], dim=1)
        num_emb = self.num_proj(num)

        # Масштабируем num_emb, чтобы не доминировали
        num_emb_scaled = num_emb * 3.0

        # Смешиваем признаки через поэлементное умножение
        ingr_image = ingr_emb * image_emb
        ingr_num = ingr_emb * num_emb_scaled
        image_num = image_emb * num_emb_scaled

        # Конкатенируем исходные и смешанные признаки
        fused = torch.cat([ingr_emb, image_emb, num_emb_scaled,
                        ingr_image, ingr_num, image_num], dim=1)



        return self.regressor(fused)

def train(config, device):
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultiModalModel(config).to(device)

    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([
        {'params': model.image_model.parameters(), 'lr': config.IMAGE_LR},
        {'params': model.num_proj.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.regressor.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.ingr_embedding.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.ingr_attention.parameters(), 'lr': config.REGRESSOR_LR},
        {'params': model.ingr_proj.parameters(), 'lr': config.REGRESSOR_LR},
    ])
    criterion = nn.L1Loss()

    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type="val")
    train_dataset = MultiModalDataset(config, transforms, type="train")
    val_dataset = MultiModalDataset(config, val_transforms, type="val")
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 ))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,))
    print("Обучение началось!")

    MAE_metric_train = torchmetrics.MeanAbsoluteError().to(device)
    MAE_metric_val = torchmetrics.MeanAbsoluteError().to(device)
    best_mae_val = 0.0 
    for epoch in range(config.EPOCHS):
        MAE_metric_train.reset()
        total_loss = 0.0
        model.train()
        for batch in train_loader:
            inps = {
            "ingredients_ids": batch["ingredients_ids"].to(device),
            "ingredients_mask": batch["ingredients_mask"].to(device),
            "image": batch["image"].to(device),
            "mass": batch["mass"].to(device),
            "amount_ing": batch["amount_ing"].to(device),
        }
            calories = batch['calories'].unsqueeze(1).to(device)
            optimizer.zero_grad()
            logits = model(**inps)

            loss = criterion(logits, calories)

            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
            _ = MAE_metric_train.update(logits, calories)
        train_mae = MAE_metric_train.compute().cpu().numpy()
        MAE_metric_train.reset()
        val_mae = validate(model, val_loader, device, MAE_metric_val)
        MAE_metric_val.reset()


        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae_val or best_mae_val == 0.0:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)


def validate(model, val_loader, device, mae_metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inps = {
            "ingredients_ids": batch["ingredients_ids"].to(device),
            "ingredients_mask": batch["ingredients_mask"].to(device),
            "image": batch["image"].to(device),
            "mass": batch["mass"].to(device),
            "amount_ing": batch["amount_ing"].to(device),
        }
            calories = batch['calories'].unsqueeze(1).to(device)

            logits = model(**inps)
            mae_metric.update(logits, calories)
           

    return mae_metric.compute().cpu().numpy()

