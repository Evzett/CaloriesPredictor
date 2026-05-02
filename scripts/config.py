from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import timm
import numpy as np
import pandas as pd

import torchmetrics
from transformers import AutoTokenizer

import albumentations as A



class Config:
    SEED = 42

    # Меняем на ResNet
    IMAGE_MODEL_NAME = "resnet50"  

    NUM_INGREDIENTS = 600  
    # Для ResNet можно размораживать последние блоки
    IMAGE_MODEL_UNFREEZE = "layer3|layer4"  

    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 3e-3
    EPOCHS = 60
    BATCH_SIZE = 64
    DROPOUT = 0.3
    HIDDEN_DIM = 256

    DISH_DF_PATH = "data/output.csv"
    INGRS_PATH = "data/data/ingredients.csv"
    IMAGES_PATH = 'data/data/images'
    SAVE_PATH = "best_model.pth"
