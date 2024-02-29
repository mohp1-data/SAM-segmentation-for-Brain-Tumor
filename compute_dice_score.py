
from utils.SurfaceDice import compute_dice_coefficient
import numpy as np
import monai
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nib
import os
import PIL as pil
from monai.networks.nets import UNet
import torch
from keras.models import load_model
from loss import custom_loss


def compute_dice_score(ground_truths, predicted_masks, save_path=None):
    '''
    Compute the Dice score given an array of ground truth masks and an array of predicted_masks. 
    If save_path is not None, saves the dice score as a .txt file at the save_path.

    Parameters:

    ground_truths : ndarray
        The ground truth masks. Must have shape (N, H, W, 4), where N is the number of images. The 4 represents each ROI.
    
    predicted_masks : ndarray
        The predicted masks. Must have shape (N, H, W, 4), where N is the number of images. The 4 represents each ROI.
    
    save_path : str, optional
        Optional path to save fps_per_image and total_sensitivity to.
    '''
    dice_scores = np.array([0, 0, 0]).astype(np.float64) # 0 is edema, 1 is non-enhancing, 2 is enhancing
    num_images = ground_truths.shape[0]

    # if ground_truths.shape != predicted_masks.shape:
    #     raise ValueError("Ground truths and predicted masks must have same shape!")
    # if len(ground_truths.shape) != 3 or ground_truths.shape[3] != 4:
    #     raise ValueError("Ground truths and predicted masks must have shape (N, H, W, 4). Received shape " + str(ground_truths.shape) + " instead.")

    for index in tqdm(range(num_images)):
        dice_scores[0] += compute_dice_coefficient(ground_truths[index, :, :]==1, predicted_masks[index, :, :]==1)
        dice_scores[1] += compute_dice_coefficient(ground_truths[index, :, :]==2, predicted_masks[index, :, :]==2)
        dice_scores[2] += compute_dice_coefficient(ground_truths[index, :, :]==3, predicted_masks[index, :, :]==3)
    
    dice_scores /= num_images

    if save_path is not None:
        np.savetxt(save_path, dice_scores, header="Edema, Non-enhancing, Enhancing\n", newline=", ", fmt="%f")

    return dice_scores
