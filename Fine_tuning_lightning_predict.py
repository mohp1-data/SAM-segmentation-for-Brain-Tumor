from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.SurfaceDice import compute_dice_coefficient
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform
from Fine_tuning_lightning_train import MedsamModel

medsam_model = MedsamModel.load_from_checkpoint("/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w/enhancing/checkpoints/all augment/version 14/epoch=76-step=28028.ckpt", roi=3, modalities=[0,2,3], npz_tr_path='')
medsam_model.to('cuda:1')

npz_file = np.load("/home/mohammad/MedSAM/data/Npz_files_three_modalities/flair-t1gd-t2w_/train_aug/flair-t1gd-t2w__BRATS_015_aug.npz")

img = torch.as_tensor(npz_file['imgs'][12], device='cuda:1')
gt2D = torch.as_tensor(npz_file['gts'][12] == 3, device='cuda:1')

# create bounding box
y_indices, x_indices = np.where(gt2D.detach().cpu().numpy() > 0)
x_min, x_max = np.min(x_indices), np.max(x_indices)
y_min, y_max = np.min(y_indices), np.max(y_indices)
# add perturbation to bounding box coordinates
H, W = gt2D.shape
x_min = max(0, x_min - np.random.randint(0, 20))
x_max = min(W, x_max + np.random.randint(0, 20))
y_min = max(0, y_min - np.random.randint(0, 20))
y_max = min(H, y_max + np.random.randint(0, 20))
bbox = torch.as_tensor([x_min, y_min, x_max, y_max], device='cuda:1')

medsam_model.predict_image(img, gt2D, bbox)