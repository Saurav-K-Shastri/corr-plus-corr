"""Implement the Dataset class for CNN denoiser train/eval/test.
Extends torch.utils.data.Dataset class for training and testing.

Example:
    To create a torch.utils.data.Dataloader
    from a DenoiseDataset dataset, use::
        data_loader = DataLoader(dataset).
Note:
    A .h5 file storing the dataset must be formatted as
        dataset.h5
            - key 1 : image 1
            - key 2 : image 2
            - ...
        where each image has dimensions (C, H, W).
"""

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
from torch.utils.data import Dataset
import h5py


class DenoiseDataset(Dataset):
    def __init__(self, datadir, transforms=None):
        """
        Args:
            datadir (str): path to .h5 file.
            transform: image transformation, for data augmentation.
            
        Note:
            If sigma or alpha are lists of two entries, the value used will be
            uniformly sampled from the two values.
        """
        super(DenoiseDataset, self).__init__()
        self.datadir = datadir
        self.transforms = transforms
        self.h5f = h5py.File(datadir, 'r')
        self.keys = list(self.h5f.keys())
#         h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """Return image, noise pair of the same size (C, H, W)."""

        key = self.keys[idx]
        image = torch.Tensor(self.h5f[key])

        
        if self.transforms is not None:
            image = self.transforms(image)
        

        return image
