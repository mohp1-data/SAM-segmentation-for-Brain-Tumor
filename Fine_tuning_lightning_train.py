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
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
import torch.nn as nn
import torch.nn.functional as F
from skimage import transform
from skimage import io, segmentation
from glob import glob

from skimage.transform import rotate
from skimage.util import img_as_ubyte
from skimage.io import imsave
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
from skimage.transform import AffineTransform, warp
from utils.SurfaceDice import compute_dice_coefficient
import imageio.v2 as iio
from PIL import Image

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


# set seeds
torch.manual_seed(2023)
np.random.seed(2023)
torch.set_float32_matmul_precision("high")  # for speed

# In[2]:


from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR


# Elastic deformation function
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        )
        * alpha
    )

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")

    # Handle both grayscale (2D) and RGB (3D) images
    if len(shape) == 2:  # if image is grayscale
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)
    else:  # if image is RGB
        # separate channels and apply transformation to each
        transformed_images = []
        for channel in range(shape[2]):
            indices = np.reshape(x + dx[:, :, channel], (-1, 1)), np.reshape(
                y + dy[:, :, channel], (-1, 1)
            )
            distorted_image = map_coordinates(
                image[:, :, channel], indices, order=1, mode="reflect"
            )
            transformed_images.append(
                distorted_image.reshape(image[:, :, channel].shape)
            )

        # stack the transformed channels back into a single image
        return np.dstack(transformed_images)


# show bounding box on image
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


# %% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset):
    def __init__(self, data_root, roi, modalities, medsam_model):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in tqdm(self.npz_files)]
        self.roi = roi

        # this implementation is ugly but it works (and is also fast for feeding data to GPU) if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        print(np.max(self.npz_data[0]["gts"]), self.npz_data[0]["imgs"].shape)
        self.ori_gts = np.vstack([d["gts"] for d in tqdm(self.npz_data)])
        self.imgs = np.vstack([d["imgs"] for d in tqdm(self.npz_data)])

        if modalities == [0, 1, 2]:
            self.img_embeddings = np.vstack(
                [d["img_embeddings"] for d in tqdm(self.npz_data)]
            )
        else:
            self.imgs = self.imgs[:, :, :, modalities]
            # recompute image embeddings for permuted RGB channel
            embeddings = []

            if len(modalities) == 3:
                for img in tqdm(self.imgs):
                    # img_np = img.detach().cpu().numpy()
                    # resize_img = medsam_model.sam_trans.apply_image(img_np)
                    # resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(img)
                    # input_image = medsam_model.sam_model.preprocess(resize_img_tensor[None, :, :, :])
                    # assert input_image.shape == (
                    #     1,
                    #     3,
                    #     medsam_model.image_encoder.img_size,
                    #     medsam_model.image_encoder.img_size,
                    # ), "input image should be resized to 1024*1024"
                    # with torch.no_grad():
                    #     embedding = medsam_model.sam_model.image_encoder(input_image)[0]
                    embedding = medsam_model.compute_image_embedding(img)
                    embeddings.append(embedding.detach().cpu().numpy())

            elif len(modalities) == 1:
                self.imgs = np.repeat(self.imgs, 3, axis=-1)
                print(self.imgs.shape)
                for img in tqdm(self.imgs):
                    embedding = medsam_model.compute_image_embedding(
                        torch.as_tensor(
                            img,
                            dtype=torch.uint8,
                            device=medsam_model.device,
                        )
                    )
                    embeddings.append(embedding.detach().cpu().numpy())
            else:
                raise ValueError("Modalities should be length 1 or 3.")

            self.img_embeddings = np.array(embeddings)

        if self.roi == "whole":
            self.ori_gts = self.ori_gts != 0
        else:
            self.ori_gts = self.ori_gts == self.roi

        print(np.max(self.ori_gts))
        print(
            f"self.imgs.shape: {self.imgs.shape}, self.img_embeddings.shape: {self.img_embeddings.shape}, self.ori_gts.shape: {self.ori_gts.shape}"
        )
        # (N, 256, 256, 3), (N, 256, 64, 64), (N, 256, 256)

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        gt = self.ori_gts[index]
        img_embed = self.img_embeddings[index]

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return (
            torch.tensor(img).to(torch.float32),
            torch.tensor(img_embed).to(torch.float32),
            torch.tensor(gt[None, :, :]).to(torch.int64),
            torch.tensor(bboxes).to(torch.float32),
        )
        # return torch.tensor(img).to(torch.float32), torch.tensor(gt[None, :,:]).to(torch.int64), torch.tensor(bboxes).to(torch.float32)


# this dataset class is for the test dataset
class NpzTestDataset(Dataset):
    def __init__(self, data_root, roi, modalities):
        self.roi = roi
        self.modalities = modalities
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in tqdm(self.npz_files)]

        # we need to manually grab the correct roi and modality
        gts_array, imgs = [], []
        for npz_file in tqdm(self.npz_data):
            # grab the correct roi from ground_truth
            print("Number of ground truth labels:", np.sum(npz_file["gts"] != 0))
            print(
                "Ground truth labels:", np.min(npz_file["gts"]), np.max(npz_file["gts"])
            )
            if self.roi == "whole":
                ground_truth = (npz_file["gts"] != 0).astype(np.uint8)
            else:
                ground_truth = (npz_file["gts"] == self.roi).astype(np.uint8)
            print("Number of selected ROI labels:", np.sum(ground_truth))
            # ground_truth = (npz_file["gts"] != 0).astype(np.uint8)
            # grab the correct modalities from the image
            image = npz_file["imgs"][:, :, :, self.modalities]
            # print(image.shape)
            if image.shape[3] == 1:  # repeated channels
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[3] != 3:
                raise ValueError("Modalities should be length 1 or 3.")

            # for each slice, apply preprocessing steps, resize to (256, 256), to input to SAM
            for i in range(ground_truth.shape[0]):
                gt2D = transform.resize(
                    ground_truth[i],
                    (256, 256),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )

                # nii preprocess start
                # lower_bound, upper_bound = np.percentile(image[i], 0.5), np.percentile(image[i], 99.5)
                # image_data_pre = np.clip(image[i], lower_bound, upper_bound)
                # image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
                # image_data_pre[image[i]==0] = 0
                # image_data_pre = image_data_pre.astype(np.uint8)

                image2D = image[i, :, :, :]
                # print(image2D.shape)
                resized_image = []

                # normalize intensities
                lower_bound, upper_bound = np.percentile(image2D, 0.5), np.percentile(
                    image2D, 99.5
                )
                for channel in range(3):
                    # normalize each modality separately
                    image_data_channel = image2D[:, :, channel]
                    image_data_pre = np.clip(
                        image_data_channel, lower_bound, upper_bound
                    )
                    image_data_pre = (
                        (image_data_pre - np.min(image_data_pre))
                        / (np.max(image_data_pre) - np.min(image_data_pre))
                        * 255.0
                    )
                    image_data_pre[image_data_channel == 0] = 0
                    image_data_pre = image_data_pre.astype(np.uint8)

                    image_data_pre = transform.resize(
                        image_data_pre,
                        (256, 256),
                        order=3,
                        preserve_range=True,
                        mode="constant",
                        anti_aliasing=True,
                    )

                    image_data_pre = np.rot90(image_data_pre, 3, (0, 1))
                    resized_image.append(image_data_pre)

                # rotate to correct orientation
                gt2D = np.rot90(gt2D, 3, (0, 1))

                gts_array.append(gt2D)
                # if (
                #     len(self.modalities) == 1
                # ):  # if only one modality, stack it 3 times in RGB channel
                #     resized_image = [
                #         resized_image[0],
                #         resized_image[0],
                #         resized_image[0],
                #     ]
                imgs.append(np.stack(resized_image, axis=-1))
                # print("Image:", len(resized_image))

        self.gts = np.array(gts_array)
        self.imgs = np.array(imgs)
        print(f"self.imgs.shape: {self.imgs.shape}, self.gts.shape: {self.gts.shape}")

    def __len__(self):
        return self.gts.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        gt2D = self.gts[index]
        y_indices, x_indices = np.where(gt2D > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            x_min, y_min = 0, 0
            x_max, y_max = (
                img.shape[1],
                img.shape[0],
            )  # full size bounding box for slices that do not have ground truth. Mohammad
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bboxes = np.array([x_min, y_min, x_max, y_max])
            # convert img embedding, mask, bounding box to torch tensor
            return (
                torch.tensor(img).to(torch.float32),
                torch.tensor(gt2D).to(torch.int64),
                torch.tensor(bboxes).to(torch.float32),
            )


class MedsamModel(pl.LightningModule):
    def __init__(self, roi, modalities, npz_tr_path, png_save_path, lr=1e-4):
        super().__init__()

        self.npz_tr_path = npz_tr_path
        self.work_dir = "./work_dir"
        self.task_name = "CT_Abd-Gallbladder"

        self.npz_test_path = "/home/mohammad/MedSAM/data/test_npz_files"
        self.roi = roi
        self.modalities = modalities

        # data augmentation parameters
        self.flip_direction = 1  # Flip image in the horizontal direction
        self.alpha = 34  # Scaling factor for elastic transform
        self.sigma = 4  # Standard deviation for Gaussian smoothing in elastic transform
        self.shear_factor = 0.1  # Shearing factor

        # prepare SAM model
        self.model_type = "vit_b"
        self.checkpoint = "work_dir/SAM/sam_vit_b_01ec64.pth"

        model_save_path = join(self.work_dir, self.task_name)
        os.makedirs(model_save_path, exist_ok=True)
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam_model.train()

        self.image_encoder = self.sam_model.image_encoder
        self.mask_decoder = self.sam_model.mask_decoder
        self.prompt_encoder = self.sam_model.prompt_encoder
        self.sam_trans = ResizeLongestSide(self.image_encoder.img_size)

        # prepare png save path
        self.png_save_path = png_save_path

        # hyperparameters
        self.learning_rate = lr

        self.seg_loss = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        self.dsc_loss = monai.losses.DiceLoss(sigmoid=True)
        self.dice_scores = []
        self.hausdorff_distances = []

        self.training_epoch_losses = []
        self.training_epoch_dsc = []

        # freeze prompt encoder and image encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.example_input_array = (
            torch.as_tensor(
                np.load(
                    "/home/mohammad/MedSAM/data/Npz_files_three_modalities/flair-t1gd-t2w_/train/flair-t1gd-t2w__BRATS_001.npz"
                )["img_embeddings"][0]
            ),
            torch.rand((256, 256)) > 0.5,
            torch.as_tensor([0, 0, 255, 255]),
        )

    @torch.jit.script_if_tracing
    @torch.jit.unused
    def apply_boxes(self, bbox, ground_truth):
        # sam_trans = ResizeLongestSide(self.image_encoder.img_size)
        bbox_np = self.sam_trans.apply_boxes(
            bbox, (ground_truth.size(dim=-2), ground_truth.size(dim=-1))
        )
        if len(bbox_np.shape) == 2:
            bbox_np = bbox_np[:, None, :]  # (B, 1, 4)
        return bbox_np

    @torch.jit.script_if_tracing
    @torch.jit.unused
    def predict_mask(
        self, image_embedding, sparse_embeddings, dense_embeddings, multimask_output
    ):
        mask_predictions, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=multimask_output,
        )

        return mask_predictions

    def forward(self, image_embedding, gt2D, bbox):
        with torch.no_grad():
            # convert box to 1024x1024 grid
            # bbox_np = bbox.detach().cpu().numpy()
            # bbox_np = self.sam_trans.apply_boxes(bbox_np, (old_h, old_w))
            bbox_np = self.apply_boxes(bbox.detach().cpu().numpy(), gt2D)
            bbox_torch = torch.from_numpy(bbox_np).to(bbox)

            # get prompt embeddings
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=bbox_torch,
                masks=None,
            )

        return self.predict_mask(
            image_embedding, sparse_embeddings, dense_embeddings, multimask_output=False
        )

    def augment(self, img, gt, embedding):
        # convert to numpy array for image manipulation
        img_np = img.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        is_rotate = np.random.random() < 0.5
        is_elastic = np.random.random() < 0.5

        if is_rotate:
            # Random rotation within -20 and +20 degrees
            angle = np.random.uniform(-20, 20)
            img_np = rotate(img_np, angle=angle, mode="reflect")
            gt_np = rotate(gt_np, angle=angle, mode="reflect")

        if np.random.random() < 0.0:
            # Flip images
            img_np = np.flip(img_np, axis=self.flip_direction)
            gt_np = np.flip(gt_np, axis=self.flip_direction)

        if is_elastic:
            # Apply elastic transform
            img_np = elastic_transform(img_np, alpha=self.alpha, sigma=self.sigma)
            gt_np = elastic_transform(gt_np, alpha=self.alpha, sigma=self.sigma)

        if np.random.random() < 0.0:
            # Apply shear transform
            shear_transform = AffineTransform(shear=self.shear_factor)
            img_np = warp(img_np, shear_transform.inverse, mode="reflect")
            gt_np = warp(gt_np, shear_transform.inverse, mode="reflect")

        # Convert to 8-bit unsigned integers
        img_np = img_np.astype(np.uint8)

        # after augmentation, the 1's in the ground truth become as small as 1e-19
        gt_np = gt_np != 0

        # recompute image embeddings for augmented image
        # resize_img = self.sam_trans.apply_image(img_np)
        # resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(img)
        # input_image = self.sam_model.preprocess(resize_img_tensor[None, :, :, :])
        # assert input_image.shape == (
        #     1,
        #     3,
        #     self.image_encoder.img_size,
        #     self.image_encoder.img_size,
        # ), "input image should be resized to 1024*1024"
        # with torch.no_grad():
        #     embedding = self.sam_model.image_encoder(input_image)[0]
        if is_rotate or is_elastic:
            new_embedding = self.compute_image_embedding(
                torch.as_tensor(img_np, device=img.device)
            )
        else:  # if not augmented at all, we can just use the original embedding
            new_embedding = embedding

        # save an example image for sanity check
        bd = segmentation.find_boundaries(gt_np, mode="inner")
        img_idx = np.copy(img_np)
        img_idx[bd, :] = [255, 0, 0]
        io.imsave(
            join(
                "/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w",
                "aug.png",
            ),
            img_idx,
            check_contrast=False,
        )

        # recompute the bounding box
        y_indices, x_indices = np.where(gt_np > 0)
        # convert to numpy array
        # y_indices = y_indices.detach().cpu().numpy()
        # x_indices = x_indices.detach().cpu().numpy()

        # The issue should be fixed.
        # What happened was in self.augment, the ground truth would be converted to float64,
        # and the 1's representing True became 1e-19.
        # When converting back to uint8, they would be rounded down to 0, causing the ground truth to become all 0.
        if y_indices.size == 0 or x_indices.size == 0:
            raise RuntimeError(
                "Ground truth empty. Perhaps the data augmentation failed."
            )
            x_min, y_min = (
                0,
                0,
            )  # handling when GT is 0 to put bounding box as whole image, Mohammad
            x_max, y_max = image.shape[1], image.shape[0]
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt_np.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            box = torch.as_tensor([x_min, y_min, x_max, y_max], device=gt.device)

        return (
            torch.as_tensor(img_np, device=img.device),
            new_embedding,
            torch.as_tensor(gt_np, device=gt.device),
            box,
        )

    def training_step(self, batch, batch_idx):
        tensorboard_logger = self.logger.experiment

        images, image_embeddings, gt2Ds, boxes = batch
        # images, gt2Ds, boxes = batch
        batch_size = images.shape[0]
        loss, dsc = 0, 0
        for index in range(batch_size):
            box = boxes[index]  # intializing the box ,Mohammad
            image = images[index]  # (256, 256, 3)
            image.to(torch.uint8)
            gt2D = gt2Ds[index].squeeze()

            is_augmented = True  # always augment data
            if is_augmented:
                image, image_embedding, gt2D, box = self.augment(
                    image, gt2D, image_embeddings[index]
                )  # now augment has Tensor input and Tensor output, Michael

            else:
                # raise RuntimeError("This should not happen. The data is not being augmented with 100 percent chance.")
                image_embedding = image_embeddings[index]
                box = boxes[index]

            # visualize image embeddings
            # tensorboard_logger.add_embedding(image_embedding[None], global_step = trainer.global_step, label_img = image.view(3, 256, 256)[None, :, :, :])

            mask_prediction = self(image_embedding, gt2D, box)
            loss += self.seg_loss(mask_prediction, gt2D[None, None, :, :])
            # with torch.no_grad():
            #     dsc += 1 - self.dsc_loss(mask_prediction, gt2D[None, None, :, :])
            dsc_single_image = compute_dice_coefficient(
                gt2D.detach().cpu().numpy(),
                (torch.sigmoid(mask_prediction) > 0.5).detach().cpu().numpy(),
            )
            dsc += dsc_single_image

            # save an image of the ground truth mask and predicted mask
            img_idx = image.detach().cpu().numpy()
            img_idx //= 2  # make the brain darker so the masks show up better

            gt_idx = gt2D.detach().cpu().numpy()

            pred_idx = torch.sigmoid(mask_prediction)[0, 0, :, :]
            pred_idx = pred_idx.detach().cpu().numpy() > 0.5

            tp = np.logical_and(pred_idx, gt_idx)
            fn = np.logical_and(np.logical_not(pred_idx), gt_idx)
            fp = np.logical_and(pred_idx, np.logical_not(gt_idx))

            gt_border = segmentation.find_boundaries(gt_idx, mode="inner")

            img_idx[tp, :] = [0, 255, 0]  # this makes true positives green
            img_idx[fn, :] = [255, 192, 203]  # mark the false negatives with pink
            img_idx[fp, :] = [255, 255, 0]  # mark false positives with yellow
            img_idx[gt_border, :] = [255, 0, 0]  # mark the ground truth border with red

            # add DSC score to image
            fig, axs = plt.subplots(
                1,
                1,
            )
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
            axs.imshow(img_idx.astype(np.uint8))
            axs.text(
                1, 10, f"DSC: {100*dsc_single_image:5.2f}", fontsize=16, color="white"
            )
            axs.axis("off")

            fig.tight_layout()
            plt.savefig(
                "/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w/train.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # io.imsave(
            #     "/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w/train.png",
            #     img_idx.astype(np.uint8),
            #     check_contrast=False,
            # )

        # find average loss and dsc for this batch
        loss /= batch_size
        dsc /= batch_size

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "DSC",
            dsc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        tensorboard_logger.add_scalar("train_loss", loss, global_step=self.global_step)
        tensorboard_logger.add_scalar("DSC", dsc, global_step=self.global_step)

        # print(type(self.training_epoch_losses), type(loss))
        self.training_epoch_losses.append(loss.item())
        self.training_epoch_dsc.append(dsc.item())
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_scalar(
            "lr", self.learning_rate, global_step=self.current_epoch
        )
        train_loss_epoch = np.mean(self.training_epoch_losses)
        train_dsc_epoch = np.mean(self.training_epoch_dsc)
        tensorboard_logger.add_scalar(
            "train_loss_epoch", train_loss_epoch, global_step=self.current_epoch
        )
        tensorboard_logger.add_scalar(
            "train_dsc_epoch", train_dsc_epoch, global_step=self.current_epoch
        )

        # free up the memory
        self.training_epoch_losses.clear()
        self.training_epoch_dsc.clear()

    def on_train_start(self) -> None:
        self.sam_model.train()
        return super().on_train_start()

    def on_test_epoch_start(self) -> None:
        self.sam_model.eval()
        self.dice_scores = []
        self.hausdorff_distances = []
        print("Mask threshold:", self.sam_model.mask_threshold)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # predict the segmentation mask using the fine-tuned model
        imgs, gt2Ds, bboxs = batch
        batch_size = imgs.shape[0]
        dsc = 0  # Dice similarity coefficient
        hd95 = 0  # Hausdorff 95 metric

        for index in range(batch_size):
            img = imgs[index]
            gt2D = gt2Ds[index]
            bbox = bboxs[index]

            # image_embedding = self.compute_image_embedding(img)
            # resize_img = self.sam_trans.apply_image(img_np)
            # resize_img_tensor = torch.from_numpy(resize_img.transpose(2, 0, 1)).to(img)
            # input_image = self.sam_model.preprocess(
            #     resize_img_tensor[None, :, :, :]
            # )  # (1, 3, 1024, 1024)
            # # input_image = self.sam_model.preprocess(img[None,:,:,:]) # (1, 3, 1024, 1024)
            # with torch.no_grad():
            #     # compute image embedding
            #     image_embedding = self.sam_model.image_encoder(
            #         input_image
            #     )  # (1, 256, 64, 64)
            # with torch.no_grad():
            #     # compute mask and apply sigmoid to convert model output to probabilities
            #     medsam_seg_prob = torch.sigmoid(self(image_embedding, gt2D, bbox))

            #     medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            #     # print("Mask threshold:", self.sam_model.mask_threshold)
            #     medsam_seg = medsam_seg_prob > 0.5

            #     # calculate DSC
            #     gt2D = gt2D.detach().cpu().numpy()
            #     dice_score = compute_dice_coefficient(gt2D, medsam_seg)
            #     self.dice_scores.append(dice_score)
            #     dsc += dice_score

            #     # calculate HD95
            #     hausdorff_distance = monai.metrics.compute_hausdorff_distance(
            #         medsam_seg[None, None, :, :],
            #         gt2D[None, None, :, :],
            #         include_background=True,
            #         percentile=95,
            #     ).item()
            #     self.hausdorff_distances.append(hausdorff_distance)
            #     hd95 += hausdorff_distance

            # save an image of the ground truth mask and predicted mask
            img_idx = img
            # gt_idx = gt2Ds.detach().cpu().numpy()[0, :, :]
            # pred_idx = medsam_seg

            # img_idx //= 2

            # tp = np.logical_and(pred_idx, gt_idx)
            # fn = np.logical_and(np.logical_not(pred_idx), gt_idx)
            # fp = np.logical_and(pred_idx, np.logical_not(gt_idx))

            # gt_border = segmentation.find_boundaries(gt_idx, mode="inner")

            # img_idx[tp, :] = [0, 255, 0]  # this makes true positives green
            # img_idx[fn, :] = [255, 192, 203]  # mark the false negatives with pink
            # img_idx[fp, :] = [255, 255, 0]  # mark false positives with yellow
            # img_idx[gt_border, :] = [
            #     255,
            #     0,
            #     0,
            # ]  # mark the ground truth border with red

            # # add DSC score to image
            # fig, axs = plt.subplots(
            #     1,
            #     1,
            # )
            # axs.axis("off")
            # plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
            # axs.imshow(img_idx.astype(np.uint8))
            # axs.text(
            #     1, 10, f"DSC: {100*dice_score:5.2f}", fontsize=16, color="white"
            # )

            # # show_box(bbox.detach().cpu().numpy(), axs)

            # fig.tight_layout()
            # plt.savefig(
            #     os.path.join(
            #         self.png_save_path,
            #         f"img-{100*dice_score:2.0f}-" + str(batch_idx) + ".png",
            #     ),
            #     bbox_inches="tight",
            #     dpi=300,
            # )
            # plt.close()
            im = Image.fromarray(img_idx.detach().cpu().numpy().astype(np.uint8))
            im.save(
                join(
                    self.png_save_path,
                    "BRATS_" + str(batch_idx).zfill(4) + "_0000.png",
                )
            )
            # io.imsave(
            #     "/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w/test.png",
            #     img_idx.astype(np.uint8),
            #     check_contrast=False,
            # )

        # take average DSC for the batch, set batch size to 1 if you want the average over all images

        # medsam_dsc /= batch_size

        # PyTorch Lightning automatically accumulates medsam_dsc and takes average over the whole epoch
        self.log_dict(
            {"DSC": dsc, "HD95": hd95},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return dsc

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.mask_decoder.parameters(), lr=self.learning_rate, weight_decay=0.0)
        # optimizer = torch.optim.SGD(self.mask_decoder.parameters(), lr=self.learning_rate, momentum=0.9)
        optimizer = torch.optim.AdamW(
            self.mask_decoder.parameters(), lr=self.learning_rate
        )
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = NpzDataset(self.npz_tr_path, self.roi, self.modalities, self)
        return DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=24
        )  # 24 is the number of cpus on this machine

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dataset = NpzTestDataset(
            self.npz_test_path, roi=self.roi, modalities=self.modalities
        )
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=24)

    def predict_image(self, img, gt2D, bbox):
        self.eval()

        # img_np = img.detach().cpu().numpy()
        # #img_np = img_np.astype(np.uint8)

        # resize_img = self.sam_trans.apply_image(img_np)
        # resize_img_tensor = torch.from_numpy(resize_img.transpose(2, 0, 1)).to(img)
        # input_image = self.sam_model.preprocess(
        #     resize_img_tensor[None, :, :, :]
        # )  # (1, 3, 1024, 1024)
        # # input_image = self.sam_model.preprocess(img[None,:,:,:]) # (1, 3, 1024, 1024)
        # with torch.no_grad():
        #     # compute image embedding
        #     image_embedding = self.sam_model.image_encoder(
        #         input_image
        #     )  # (1, 256, 64, 64)
        image_embedding = self.compute_image_embedding(img)
        with torch.no_grad():
            # compute mask and apply sigmoid to convert model output to probabilities
            medsam_seg_prob = torch.sigmoid(self(image_embedding, gt2D, bbox))

            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            # print("Mask threshold:", self.sam_model.mask_threshold)
            medsam_seg = medsam_seg_prob > 0.5

            # save an image of the ground truth mask and predicted mask
            img_idx = img.detach().cpu().numpy()
            # gt_idx = gt2D.detach().cpu().numpy()
            pred_idx = medsam_seg

            img_idx //= 2

            # tp = np.logical_and(pred_idx, gt_idx)
            # fn = np.logical_and(np.logical_not(pred_idx), gt_idx)
            # fp = np.logical_and(pred_idx, np.logical_not(gt_idx))

            # gt_border = segmentation.find_boundaries(gt_idx, mode='inner')

            # img_idx[tp, :] = [0, 255, 0] # this makes true positives green
            # img_idx[fn, :] = [255, 192, 203] # mark the false negatives with pink
            # img_idx[fp, :] = [255, 255, 0] # mark false positives with yellow
            # img_idx[gt_border, :] = [255, 0, 0] # mark the ground truth border with red
            pred_border = segmentation.find_boundaries(pred_idx, mode="inner")
            img_idx[pred_idx, :] = [173, 216, 230]
            img_idx[pred_border, :] = [0, 0, 255]
            io.imsave(
                "/home/mohammad/MedSAM/models/MedSAM Lightning/flair-t1gd-t2w/brain015-slice12.png",
                img_idx.astype(np.uint8),
                check_contrast=False,
            )

    def compute_image_embedding(self, img):
        img_np = img.detach().cpu().numpy()
        img_np = img_np.astype(np.uint8)

        resize_img = self.sam_trans.apply_image(img_np)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(img)
        input_image = self.sam_model.preprocess(resize_img_tensor[None, :, :, :])
        assert input_image.shape == (
            1,
            3,
            self.image_encoder.img_size,
            self.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        with torch.no_grad():
            embedding = self.sam_model.image_encoder(input_image)  # (1, 256, 64, 64)
            return embedding[0]  # (256, 64, 64)


class FineTuneLearningRateFinder(pl.callbacks.LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


if __name__ == "__main__":
    # %% set up model for fine-tuning
    # fmt: off
    rois_dict = {"background": 0, 
                 "edema": 1, 
                 "non-enhancing": 2, 
                 "enhancing": 3}

    modalities_dict = {"flair": 0, 
                       #"t1w": 1, 
                       "t1gd": 1, 
                       "t2w": 2}
    # the train data only contains flair, t1gd, t2w in that order

    # fmt: on
    roi = rois_dict["non-enhancing"]
    # roi = "whole"
    # modalities = [
    #     modalities_dict["flair"],
    #     modalities_dict["t1gd"],
    #     modalities_dict["t2w"],
    # ]
    modalities = [modalities_dict["flair"]]

    medsam_model = MedsamModel(
        roi=roi,
        modalities=modalities,
        npz_tr_path="/home/mohammad/MedSAM/data/Npz_files_three_modalities/flair-t1gd-t2w_/train",
        png_save_path="/home/mohammad/MedSAM/models/MedSAM Lightning",
    )
    # medsam_model.learning_rate = 1e-4

    # %% setup the model

    num_epochs = 30

    swa = pl.callbacks.StochasticWeightAveraging(
        swa_lrs=1e-4, swa_epoch_start=0.8, annealing_strategy="cos"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="/home/mohammad/MedSAM/models/MedSAM Lightning/flair/non-enhancing/checkpoints",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
        save_on_train_epoch_end=True,
    )

    # tensorboard logger

    logger = TensorBoardLogger(
        "/home/mohammad/MedSAM/models/MedSAM Lightning/flair/non-enhancing",
        log_graph=False,
        version="augment",
    )

    # learning rate finder
    learning_rate_finder = FineTuneLearningRateFinder(
        milestones=[i for i in range(0, int(0.8 * num_epochs), 5)]
    )

    # log computational graph
    # logger.log_graph(medsam_model)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=num_epochs,
        devices=[1],
        strategy="ddp_find_unused_parameters_true",
        callbacks=[checkpoint_callback, learning_rate_finder, swa],
        logger=logger,
    )

    # %%
    # find optimal learning rate
    print("Learning rate:", medsam_model.learning_rate)
    # %% train the model
    trainer.fit(medsam_model)
