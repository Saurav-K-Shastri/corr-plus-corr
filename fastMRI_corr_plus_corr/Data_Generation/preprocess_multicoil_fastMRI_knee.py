# reference implementation: https://github.com/LabForComputationalVision/bias_free_denoising/blob/master/data/preprocess_bsd400.py
# Author: Saurav (shastri.19@osu.edu)
# cmd to run: PYTHONPATH=. python 'preprocess_multicoil_fastMRI_knee.py' 

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import *
from common_fastMRI import utils_fastMRI
from common_fastMRI.utils_fastMRI import tensor_to_complex_np
from skimage.measure import compare_ssim, compare_psnr
from data_fastMRI import transforms_new
from data_fastMRI.mri_data_new import MiddleSliceData_Singlecoil
from data_fastMRI.mri_data_new import SelectiveSliceData_Train
from data_fastMRI.mri_data_new import SelectiveSliceData_Val
import scipy.misc
import copy
import pathlib
from pathlib import Path 
import xml.etree.ElementTree as ET
import h5py
import cv2
import argparse
import sigpy as sp
import sigpy.mri as mr
    
def Rss(x):
    y = np.expand_dims(np.sum(np.abs(x)**2,axis = -1)**0.5,axis = 2)
    return y

def ImageCropandKspaceCompression(x,image_size):
#     print(x.shape)
#     plt.imshow(np.abs(x[:,:,0]), origin='lower', cmap='gray')
#     plt.show()
        
    w_from = (x.shape[0] - image_size) // 2  # crop images into 320x320
    h_from = (x.shape[1] - image_size) // 2
    w_to = w_from + image_size
    h_to = h_from + image_size
    cropped_x = x[w_from:w_to, h_from:h_to,:]
    
#     print('cropped_x shape: ',cropped_x.shape)
    if cropped_x.shape[-1] >= 8:
        x_tocompression = cropped_x.reshape(image_size**2,cropped_x.shape[-1])
        U,S,Vh = np.linalg.svd(x_tocompression,full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:,0:8].reshape(image_size,image_size,8)
    else:
        coil_compressed_x = cropped_x
        
    return coil_compressed_x


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, use_seed=False):
        
        self.resolution = args.resolution
        self.scaling_mode = args.scaling_mode
        self.percentile_scale = args.percentile_scale
        self.constant_scale = args.constant_scale
        
        
    def __call__(self, kspace):

        fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
        ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

        numcoil = 8

        kspace = kspace.transpose(1,2,0) 

        x = ifft(kspace, (0,1)) #(768, 396, 16)

        coil_compressed_x = ImageCropandKspaceCompression(x,self.resolution) #(384, 384, 8)

        RSS_x = np.squeeze(Rss(coil_compressed_x))# (384, 384)

        kspace = fft(coil_compressed_x, (1,0)) #(384, 384, 8)

        kspace = transforms_new.to_tensor(kspace)

        kspace = kspace.permute(2,0,1,3)
 
        kspace_np = tensor_to_complex_np(kspace)

        ESPIRiT_tresh = 0.02
#         ESPIRiT_crop = 0.96
#         ESPIRiT_crop = 0.95
        ESPIRiT_crop = 0
        ESPIRiT_width_full = 24
        ESPIRiT_width_mask = 24
        device=sp.Device(-1)
        
        sens_maps = mr.app.EspiritCalib(kspace_np,calib_width= ESPIRiT_width_mask,thresh=ESPIRiT_tresh, kernel_width=6, crop=ESPIRiT_crop,device=device,show_pbar=False).run()
        
        sens_maps = sp.to_device(sens_maps, -1)
        
        sens_map_foo = np.zeros((self.resolution,self.resolution,8)).astype(np.complex128)
        
        for  i in range(8):
            sens_map_foo[:,:,i] = sens_maps[i,:,:]

        lsq_gt = np.sum(sens_map_foo.conj()*coil_compressed_x , axis = -1)   
        
        image = transforms_new.to_tensor(lsq_gt)
        
        if self.scaling_mode == 'percentile':
            sorted_image_vec = transforms_new.complex_abs(image).reshape((-1,)).sort()
            scale = sorted_image_vec.values[int(len(sorted_image_vec.values) * self.percentile_scale/100)].item()
        elif self.scaling_mode == 'absolute_max':
            scale = transforms_new.complex_abs(image).max()
        else:
            scale = self.constant_scale #  number obtained by taking the mean value of the max values of the training images (check Single_coil_Knee_Data_Access.ipynb)
        
#         scale = 0.0012 # constant scale, This is the scaling Ted used in his codes. 
        
#         image_abs = transforms_new.complex_abs(image)
        
        image = image/scale
        image = image.permute(2,0,1)
        
        return image.numpy()
    
    

def create_test_datasets(args):

    train_data = SelectiveSliceData_Train(
        root=args.data_path_train,
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True, # Set true. It actually uses mid slices
        number_of_top_slices=8, #Although it says top slices, the code uses mid 8 slices
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )

    val_data = SelectiveSliceData_Val(
        root=args.data_path_val,
        transform=DataTransform(args),
        challenge='multicoil',
        sample_rate=1,
        use_top_slices=True, # Set true. It actually uses mid slices
        number_of_top_slices=4, #Although it says top slices, the code uses mid 8 slices
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )
    
    return train_data, val_data


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 0 + 1 : stride, 0 : endh - win + 0 + 1 : stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def main(args):
    
    train_data, val_data = create_test_datasets(args)
        
    print(" ")
    print("Processing training data")

    scales = [1] # [1, 0.9, 0.8, 0.7]

    with h5py.File(os.path.join(args.data_save_path, "train.h5"), "w") as h5f:
        train_size = 0
        for train_idx in range(len(train_data)):
            if train_idx%20 == 0:
                print(train_idx, '/', len(train_data))
            img = np.transpose(train_data[train_idx], (1,2,0))
            h, w, c = img.shape
            for k in range(len(scales)):
                Img = cv2.resize(img, (int(w * scales[k]),int(h * scales[k])), interpolation=cv2.INTER_CUBIC)
                Img = np.transpose(Img, (2,0,1))
                patches = Im2Patch(Img, win=args.patch_size, stride=args.stride)
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset(str(train_size), data=data)
                    train_size += 1
                    for m in range(args.aug_times - 1):
                        data_aug = data_augmentation(data, np.random.randint(1, 8))
                        h5f.create_dataset(str(train_size) + "_aug_%d" % (m + 1), data=data_aug)
                        train_size += 1
                        
    h5f.close()
    
    print(" ")
    print("Processing validation data")
    
    with h5py.File(os.path.join(args.data_save_path, "valid.h5"), "w") as h5f:
        valid_size = 0
        for val_idx in range(len(val_data)):
            if val_idx%10 == 0:
                print(val_idx, '/', len(val_data))
            img = val_data[val_idx]
            h5f.create_dataset(str(valid_size), data=img)
            valid_size += 1
            
    h5f.close()
    
    print(" ")
    print(f"Training size {train_size}, validation size {valid_size}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-path-train", default='/storage/fastMRI/data/multicoil_knee/train', help="path to data directory for training") # change this based on your data directory
    parser.add_argument("--data-path-val", default='/storage/fastMRI/data/multicoil_knee/val', help="path to data directory for validation") # change this based on your data directory
    parser.add_argument("--data-save-path", default='/storage/fastMRI/data/multicoil_knee/pre_processed/', help="path to data directory")# change this based on where you want to save the files
    parser.add_argument("--patch-size", default=64, help="patch size")
    parser.add_argument("--resolution", default=368, help="resolution size for validation")
    parser.add_argument("--stride", default=10, help="stride")
    parser.add_argument("--aug-times", default=3, help="number of augmentations")
    parser.add_argument('--scaling-mode', type=str, default='percentile', help='scaling mode (percentile, absolute_max, constant)')
    parser.add_argument('--percentile-scale', type=float, default=98, help='percentile-scale for percentile scaling mode')
    parser.add_argument('--constant-scale', type = float, default=0.0012, help='constant-scale for constant scaling mode')
    args = parser.parse_args()
    
    main(args)