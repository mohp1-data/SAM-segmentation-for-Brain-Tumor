#%%
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



def threshold(num, thresh):
    if num > thresh:
        return num
    return 0
threshold_v = np.vectorize(threshold, otypes=[np.float64])




def plot_probability_histogram(predicted_masks, model_name = None, n_bins = 1000, x_bound = [1e-5, 1], y_bound = [1e3, 1e8]):
    fig, axs = plt.subplots(1, 1, tight_layout=True)

    if model_name != None:
        axs.set_title(model_name)

    axs.set_xscale("log")
    axs.set_yscale("log")
    
    axs.set_xlabel("Probability Value")
    axs.set_ylabel("Number of Points")
    
    axs.set_xlim(x_bound[0], x_bound[1])
    axs.set_ylim(y_bound[0], y_bound[1])

    axs.hist(predicted_masks.flatten(), bins=n_bins)
    plt.show()
    plt.close()



def compute_froc_curve(ground_truths, predicted_masks, save_path=None):
    '''
    Compute the FROC curve given an array of ground truth masks and an array of predicted_masks. 
    If save_path is not None, saves the fps_per_image and total_sensitivity as a .npz at the save_path.

    Parameters:

    ground_truths : ndarray
        The ground truth masks. Must have shape (N, H, W), where N is the number of images.
    
    predicted_masks : ndarray
        The predicted masks. Must have shape (N, H, W), where N is the number of images.
    
    save_path : str, optional
        Optional path to save fps_per_image and total_sensitivity.
    '''

    fp_probs = np.array([])
    tp_probs = np.array([])
    num_targets = 0
    for i, predicted_mask in tqdm(enumerate(predicted_masks)):
        # grab nonzero mask points and their x- and y- coordinates
        mask_points = np.nonzero(predicted_mask)
        probs = np.array([predicted_mask[mask_points[0][i]][mask_points[1][i]] for i in range(len(mask_points[0]))])
        x_coords = mask_points[1]
        y_coords = mask_points[0]

        
        fp_prob, tp_prob, n = monai.metrics.compute_fp_tp_probs(probs, y_coords, x_coords, ground_truths[i])
        fp_probs = np.concatenate((fp_probs, fp_prob), axis = 0)
        tp_probs = np.concatenate((tp_probs, tp_prob), axis = 0)

        num_targets += n

    print("Fp probs and Tp probs shape:", fp_probs.shape, tp_probs.shape)

    fps_per_image, total_sensitivity = monai.metrics.compute_froc_curve_data(fp_probs, tp_probs, num_targets, len(ground_truths))

    fig, ax = plt.subplots(1,1)
    ax.plot(fps_per_image, total_sensitivity, label="HEY")
    ax.set_xlim([0, 8])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positives Per Image')
    ax.set_ylabel('Sensitivity')
    ax.set_title('FROC Curves')
    ax.legend(loc="lower right")
    plt.show()
    plt.close()

    # save fps_per_image, total_sensitivity as .npz
    if save_path != None:
        np.savez(save_path, fps_per_image=fps_per_image, total_sensitivity=total_sensitivity)

#%%
if __name__ == "__main__":
    save_path = "/home/mohammad/MedSAM/FROC curve/Attention 2D UNet"
    path_to_test_files = "/home/mohammad/MedSAM/data/Npz_files_2/MRI_in/test/*"
    path_to_model = "/home/mohammad/MedSAM/models/Attention 2D UNet/model3_with_swa.hdf5"

    # get ground truth masks
    # ground_truths = []
    # test_npzs = glob(path_to_ground_truths)
    # for npz_ts_path in test_npzs:
    #     npz = np.load(npz_ts_path)
    #     for mask in npz['gts']:
    #         ground_truths.append(mask)

    # ground_truths = np.array(ground_truths)
    # print(ground_truths.shape)

    #%%
    # load .npz files
    npzs = glob(path_to_test_files)
    npzs.sort()
    print("Number of Brains:", len(npzs))

    ground_truths = []
    images = []
    # train_ids = []

    # for npz in npzs:
        # print(int(os.path.basename(npz).split('_')[3][:3]))
        # train_ids.append(int(os.path.basename(npz).split('_')[3][:3]))

    # train_ids = np.array(train_ids)
    # print(train_ids.shape)
    # np.savetxt("/home/mohammad/MedSAM/train_ids.txt", train_ids, newline=',', fmt='%d')

    for npz in tqdm(npzs):
        with np.load(npz) as data:
            gts = data["gts"]
            imgs = data["imgs"]
            assert len(imgs) == len(gts), "Ground truth and images must have same length!"

            for i in range(len(gts)):
                ground_truths.append(gts[i])
                images.append(imgs[i])
        

    print("Number of Images:", len(ground_truths))

    #%%

    # load in the model
    # model = UNet(spatial_dims=3, in_channels=4, out_channels=4, 
    #              channels=(16, 32, 64, 128, 256),strides=(2, 2, 2, 2), 
    #              num_res_units=2)

    # model.load_state_dict(torch.load(path_to_model))
    # model.eval()

    model = load_model(path_to_model, custom_objects={'custom_loss': custom_loss})

    # #%%
    # for index in tqdm(range(len(niftis))):

    #     label = np.asarray(nib.load(all_labels_dir[index]).dataobj)
    #     if np.sum(label) > 500: # only take tumors with more than 500 pixels
    #         ground_truths.append(label)
        
    #         image = np.asarray(nib.load(niftis[index]).dataobj)
    #         predicted_masks.append(image)

    # #%%
    # # format into shape (N, H, W) where N is number of images

    # predicted_masks = np.array(predicted_masks)
    # # predicted_masks = np.concatenate(predicted_masks, axis=2)
    # # predicted_masks = predicted_masks.transpose(2,0,1)
    # print(predicted_masks.shape)

    # ground_truths = np.array(ground_truths)
    # # ground_truths = np.concatenate(ground_truths, axis=2)
    # # ground_truths = ground_truths.transpose(2,0,1)
    # ground_truths = (ground_truths == 3).astype(np.uint8) # we only care about ROI 3, the enhancing tumor
    # print(ground_truths.shape)

    #%%

    ground_truths = np.array(ground_truths)

    # seems like images has shape (N, 256, 256, 3), but it is equivalent to stacking 3 identical (N, 256, 256) arrays along axis 3
    images = np.array(images) 

    # this should be 0
    print("Number of element-wise differences between image channels:", np.sum(images[:, :, :, 0] != images[:, :, :, 1]) 
                                                + np.sum(images[:,:,:,1] != images[:,:,:,2] 
                                                + np.sum(images[:, :, :, 2] != images[:, :, :, 0])))

    #images = images[:, :, :, 0]

    ground_truths.shape, images.shape

    #%%
    # predict masks using the model
    #predicted_masks = []
    predicted_masks = model.predict(np.concatenate((images[:, 24:232, 32:224, :], images[:, 24:232, 32:224, 0:1]), axis=-1))

    #%%
    predicted_masks.shape

    #%%
    #predicted_masks = predicted_masks[:, :, :, 1]
    #%%
    plot_probability_histogram(predicted_masks, "TEST")
    #%%
    #predicted_masks = threshold_v(predicted_masks, 1e-3)
    predicted_masks = np.pad(predicted_masks, ((0,0),(24,24),(32,32),(0,0)), mode='constant')
    predicted_masks.shape

    #%%
    # 3 corresponds to enhancing tumor region

    compute_froc_curve(ground_truths[:, :, :], predicted_masks[:, :, :, 3])#, save_path=save_path)


    # %%
    # example image

    pred_mask = pil.Image.fromarray(255*((predicted_masks[:,:,:,3][0]).astype(np.uint8)), mode="L")
    gt_mask = pil.Image.fromarray(255*ground_truths[0], mode="L")
    original_image = pil.Image.fromarray(255*images[0,24:232,32:224,0], mode="L")
    # %%
    pred_mask

    #%%
    gt_mask
    # %%

    original_image
    # %%
