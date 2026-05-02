from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import timm
import numpy as np
import pandas as pd
import os

import torchmetrics


import albumentations as A

class MultiModalDataset(Dataset):
    def __init__(self, config, transforms, type):
        
        self.transforms = transforms
        self.type = type
        self.ingrs = pd.read_csv(config.INGRS_PATH)
        self.folder_path = config.IMAGES_PATH
        df = pd.read_csv(config.DISH_DF_PATH)
        if type == 'train':
            self.df = df[df['split'] == 'train'].reset_index(drop=True)
            self.mass_mean, self.mass_std = self.df['total_mass'].mean(), self.df['total_mass'].std()
            self.amount_mean, self.amount_std = self.df['amount_ing'].mean(), self.df['amount_ing'].std()
        elif type in ['val', 'test']:
            self.df = df[df['split'] == type].reset_index(drop=True)
            # чтобы не «подсматривать» на val/test, mean/std берем из train
            train_df = df[df['split'] == 'train']
            self.mass_mean = train_df['total_mass'].mean()
            self.mass_std = train_df['total_mass'].std()
            self.amount_mean = train_df['amount_ing'].mean()
            self.amount_std = train_df['amount_ing'].std()
        self.image_cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.loc[idx, "ingredients"].split(';')
        ingr_ids = [int(i.replace('ingr_', '0')) for i in text]
        ingr_names = self.ingrs[self.ingrs["id"].isin(ingr_ids)]


        img_path = self.df.loc[idx, "dish_id"]

        mass = self.df.loc[idx, 'total_mass']

        calories = self.df.loc[idx, 'total_calories']

        am_ingrs = self.df.loc[idx, "amount_ing"]

        mass = (mass - self.mass_mean) / self.mass_std
        amount_ing = (am_ingrs - self.amount_mean) / self.amount_std

        img_path = os.path.join(self.folder_path, f"{img_path}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            
            image = torch.randint(
                0, 255, (*self.image_cfg.input_size[1:], self.image_cfg.input_size[0])
            ).to(torch.float32)

        if self.transforms:
            image = self.transforms(image=np.array(image))["image"]

        return {
            "image": image,
            "ingredients_ids": ingr_ids,
            "mass": mass,
            "calories": calories,
            "amount_ing": amount_ing,
        }


def collate_fn(batch):
    max_len = max(len(x["ingredients_ids"]) for x in batch)

    ids = []
    mask = []

    for item in batch:
        cur = item["ingredients_ids"]
        pad = max_len - len(cur)

        ids.append(cur + [0]*pad)
        mask.append([1]*len(cur) + [0]*pad)

    return {
        "ingredients_ids": torch.tensor(ids, dtype=torch.long),
        "ingredients_mask": torch.tensor(mask, dtype=torch.float),
        "image": torch.stack([x["image"] for x in batch]),
        "mass": torch.tensor([x["mass"] for x in batch], dtype=torch.float),
        "amount_ing": torch.tensor([x["amount_ing"] for x in batch], dtype=torch.float),
        "calories": torch.tensor([x["calories"] for x in batch], dtype=torch.float),
    }

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg("resnet50")  # cfg.input_size

    if ds_type == "train":
        transforms = A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.3),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2(p=1.0)
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.ToTensorV2(p=1.0)
        ])
    return transforms
