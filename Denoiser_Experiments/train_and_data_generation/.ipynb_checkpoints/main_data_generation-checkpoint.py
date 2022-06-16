## I used the following python command to run the file: PYTHONPATH=. python 'train_and_data_generation/main_data_generation.py'
# Manually change datadir field accordingly

import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from preprocess import generate_datasets
from dataset import DenoiseDataset
import solve
import train_util
from utils.general import mkdir_if_not_exists, log_arguments, print_arguments

# Argument parsing
parser = argparse.ArgumentParser(description='Preprocess or Train/Test CNN denoiser')

# Common arguments
parser.add_argument('--datadir', type=str,default = '/storage/MRI_Stanford_2D_FSE/',
                    help='Path to images or .h5 files.')
parser.add_argument('--gpu', action='store_true',
                    help='Whether to use GPU for computations when applicable.')

# Arguments for preprocess
parser.add_argument('--numtrain', type=int, default=None,
                    help='Number of train images. If None, use all images in train directory.')
parser.add_argument('--numval', type=int, default=None,
                    help='Number of val images. If None, use all images in val directory.')
parser.add_argument('--numtest', type=int, default=None,
                    help='Number of test images. If None, use all images in test directory.')
parser.add_argument('--trainwindow', type=int, default=48,
                    help='Patch size of training images.')

# Set up device
args = parser.parse_args()
USE_GPU = args.gpu
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device: {}'.format(device))

if __name__ == '__main__':

    print('Parsing arguments...')
    datadir = args.datadir
    num_train = args.numtrain
    num_val = args.numval
    num_test = args.numtest
    train_window = args.trainwindow

    print('Generating datasets...')
    generate_datasets(datadir, num_train=num_train,
                      num_val=num_val, num_test=num_test, train_window=train_window)

    print('Generating dataset: Done!')
