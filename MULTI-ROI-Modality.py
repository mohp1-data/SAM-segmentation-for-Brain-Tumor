# depending on the modalities/ROI needed the code can be changed
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join 
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse
import torch
import torchvision.transforms as T

# set up the parser
parser = argparse.ArgumentParser(description='preprocess non-CT images')
parser.add_argument('-i', '--nii_path', type=str, default='/home/mohammad/MedSAM/data/images', help='path to the nii images')
parser.add_argument('-gt', '--gt_path', type=str, default='/home/mohammad/MedSAM/data/labels', help='path to the ground truth',)
parser.add_argument('-o', '--npz_path', type=str, default='/home/mohammad/MedSAM/data/Npz_files_all_roi_modality', help='path to save the npz files')

parser.add_argument('--image_size', type=int, default=256, help='image size')
parser.add_argument('--modality', type=str, default='MRI', help='modality')
parser.add_argument('--anatomy', type=str, default='in', help='anatomy')
parser.add_argument('--img_name_suffix', type=str, default='.nii.gz', help='image name suffix')
parser.add_argument('--label_id', nargs='+', type=int, default=[1,2,3], help='label id(s)') # changing it so that it can have multiple lable id(s)
parser.add_argument('--prefix', type=str, default='Brain_', help='prefix')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='work_dir/SAM/sam_vit_b_01ec64.pth', help='checkpoint')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
# seed
parser.add_argument('--seed', type=int, default=2023, help='random seed')
args = parser.parse_args()

prefix = args.modality + '_' + args.anatomy
names = sorted(os.listdir(args.gt_path))
names = [name for name in names if not os.path.exists(join(args.npz_path, prefix + '_' + name.split('.nii.gz')[0]+'.npz'))]
names = [name for name in names if os.path.exists(join(args.nii_path, name.split('.nii.gz')[0] + args.img_name_suffix))]


# split names into training and testing
np.random.seed(args.seed)
np.random.shuffle(names)
train_names = sorted(names[:int(len(names)*0.8)])
test_names = sorted(names[int(len(names)*0.8):])
print("Label_ids", args.label_id)
# def preprocessing function
def preprocess_nonct(gt_path, nii_path, gt_name, image_name, label_ids, image_size, sam_model, device):
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    slices = [] # this is being used for storing the slices that met the ground truth if condition

    flair = []
    t1w = []
    t1gd = []
    t2w = []
    modalities = [flair, t1w, t1gd, t2w]

    gts =  []
    flair_embeddings = []
    t1w_embeddings = []
    t1gd_embeddings = []
    t2w_embeddings = []
    modalities_embeddings = [flair_embeddings, t1w_embeddings, t1gd_embeddings, t2w_embeddings]

    img_modalities = []
    #assert np.max(gt_data)==1 and np.unique(gt_data).shape[0]==2, 'ground truth should be binary'

    img_sitk = sitk.ReadImage(join(nii_path, image_name))
    image_data = sitk.GetArrayFromImage(img_sitk)
    print(gt_data.shape, image_data.shape)
    # append all four modalities as separate images into img_modalities
    # we will save each modality as a separate image in the .npz file
    for modality in range(4):
        img_data = image_data[modality, :, :, :]
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(img_data, 0.5), np.percentile(img_data, 99.5)
        img_data_pre = np.clip(img_data, lower_bound, upper_bound)
        img_data_pre = (img_data_pre - np.min(img_data_pre))/(np.max(img_data_pre)-np.min(img_data_pre))*255.0
        img_data_pre[img_data==0] = 0
        img_data_pre = np.uint8(img_data_pre)
        img_modalities.append(img_data_pre)

    z_index, _, _ = np.where(gt_data>0)
    z_min, z_max = np.min(z_index), np.max(z_index)

    for i in range(z_min, z_max):
        gt_slice_i = gt_data[i,:,:]
        #gt_slice_i = transform.resize(gt_slice_i, (image_size, image_size), order=0, preserve_range=True, mode='constant', anti_aliasing=True)
        
        # use GPU to resize
        gt_slice_i_torch = torch.as_tensor(gt_slice_i, device=device)[None, :, :]
        gt_slice_i_torch = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST)(gt_slice_i_torch)
        gt_slice_i = gt_slice_i_torch[0].detach().cpu().numpy()

        # Check if condition is met
        valid = True
        for label_id in label_ids:
            if np.sum(gt_slice_i == label_id) <= 50:
                valid = False
                break

        if valid:
            # 2. Append the index to the list
            slices.append(i)
            
            # double check the ground truth
            assert np.sum(gt_slice_i)>40, 'ground truth should have more than 40 pixels'
            gts.append(gt_slice_i)

            # save each image modality
            for modality, img in enumerate(img_modalities):
            # resize img_slice_i to 256x256, using GPU
                #img_slice_i = transform.resize(img[i,:,:], (image_size, image_size), order=3, preserve_range=True, mode='constant', anti_aliasing=True) # adjusted here
                img_slice_i = img[i, :, :]

                img_slice_i_torch = torch.as_tensor(img_slice_i, device=device)[None, :, :]
                img_slice_i_torch = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(img_slice_i_torch)
                img_slice_i = img_slice_i_torch[0].detach().cpu().numpy()

                # convert to three channels
                img_slice_i = np.uint8(np.repeat(img_slice_i[:,:,None], 3, axis=-1))
                
                assert len(img_slice_i.shape)==3 and img_slice_i.shape[2]==3, 'image should be 3 channels'
                assert img_slice_i.shape[0]==gt_slice_i.shape[0] and img_slice_i.shape[1]==gt_slice_i.shape[1], 'image and ground truth should have the same size'
                
                modalities[modality].append(img_slice_i)
                
                
                
                

                # compute image embeddings for each modality
                if sam_model is not None:
                    sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                    resize_img = sam_transform.apply_image(img_slice_i)
                    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
                    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:])
                    
                    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
                    
                    with torch.no_grad():
                        embedding = sam_model.image_encoder(input_image)
                        modalities_embeddings[modality].append(embedding.cpu().numpy()[0])
                        

    if sam_model is not None:
        return modalities, gts, modalities_embeddings, slices
    else:
        return modalities, gts, None, slices



#%% prepare the save path
save_path_tr = join(args.npz_path, prefix, 'train')
os.makedirs(save_path_tr, exist_ok=True)

# don't need test path

#save_path_ts = join(args.npz_path, prefix, 'test')
#os.makedirs(save_path_ts, exist_ok=True)

#%% set up the model

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)

for name in tqdm(train_names):
    image_name = name.split('.nii.gz')[0] + args.img_name_suffix
    gt_name = name 

    imgs, gts, img_embeddings, slices = preprocess_nonct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model, args.device)
    
    # save to npz file
    # stack the list to array
    if len(imgs[0]) >= 1:
        imgs = np.stack(imgs, axis=0) # (4, n, 256, 256, 3)
        gts = np.stack(gts, axis=0) # (n, 256, 256)

        if img_embeddings is not None:
            img_embeddings = np.stack(img_embeddings, axis=0) # (4, n, 256, 64, 64)

            print(imgs.shape, gts.shape, img_embeddings.shape)
            np.savez_compressed(join(save_path_tr, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'),
                                imgs=imgs, gts=gts, img_embeddings=img_embeddings)
            print("Saved!")
        
        else:
            print(imgs.shape, gts.shape, "No image embeddings")
        
        
        
        # save example images for sanity check
        for modality in range(4):
            idx = np.random.randint(0, imgs[modality].shape[0])
            img_idx = imgs[modality, idx,:,:,:]
            gt_idx = gts[idx,:,:]
            bd = segmentation.find_boundaries(gt_idx, mode='inner')
            img_idx[bd, :] = [255, 0, 0]
            io.imsave(save_path_tr + str(modality) + '.png', img_idx, check_contrast=False)
        




# save testing data
# for name in tqdm(test_names):
#     image_name = name.split('.nii.gz')[0] + args.img_name_suffix
#     gt_name = name 
#     imgs, gts, _ = preprocess_nonct(args.gt_path, args.nii_path, gt_name, image_name, args.label_id, args.image_size, sam_model=None, device=args.device)
#     #%% save to npz file
#     if len(imgs)>1:
#         imgs = np.stack(imgs, axis=0) # (n, 256, 256, 3)
#         gts = np.stack(gts, axis=0) # (n, 256, 256)
#         img_embeddings = np.stack(img_embeddings, axis=0) # (n, 1, 256, 64, 64)
#         np.savez_compressed(join(save_path_ts, prefix + '_' + gt_name.split('.nii.gz')[0]+'.npz'), imgs=imgs, gts=gts)
#         # save an example image for sanity check
#         idx = np.random.randint(0, imgs.shape[0])
#         img_idx = imgs[idx,:,:,:]
#         gt_idx = gts[idx,:,:]
#         bd = segmentation.find_boundaries(gt_idx, mode='inner')
#         img_idx[bd, :] = [255, 0, 0]
#         io.imsave(save_path_ts + '.png', img_idx, check_contrast=False)
