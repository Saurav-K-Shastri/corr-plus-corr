## I used the following python command to run the file=: PYTHONPATH=. python 'train_and_data_generation/train_BF_CNN_image_denoiser_correlated_noise.py' --output-dir 'image_denoiser_experiments_correlated_noise'
# manually change "my_stds" values in line 96
# Manually change datadir field accordingly

import os
import argparse
import logging
import sys
import torch
import torchvision
import torch.nn.functional as F

from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from utils import data
import models, utils

from utils.general import mkdir_if_not_exists, log_arguments, print_arguments


from utils.general import mkdir_if_not_exists, log_arguments, print_arguments

from utils.noise_model import get_noisy_data_and_noise_with_same_stat

from utils.noise_model import get_noisy_data_and_stds


import train_util
import numpy as np
from dataset import DenoiseDataset

def main(args):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))
    
    utils.setup_experiment(args)
    utils.init_logging(args)
    

    # Build data loaders, a model and an optimizer
    model = models.build_model(args).to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 60, 70, 80, 90, 100], gamma=0.5)
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(403200/state_dict['args'].batch_size))+1
    else:
        global_step = -1
        start_epoch = 0
    
    print('Parsing training arguments...')
    datadir = args.datadir
    print('Parameters:')
    print_arguments(args)
    print()
    
    print('Setting up training...')
    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        train_util.FixedAngleRotation([0, 90, 180, 270]),
        transforms.ToTensor(),
    ])
    
#     dataset_train = DenoiseDataset(os.path.join(datadir, 'train.h5'))
    dataset_train = DenoiseDataset(os.path.join(datadir, 'train.h5'), transforms=transforms_train)
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = DenoiseDataset(os.path.join(datadir, 'val.h5'))
    valid_loader = DataLoader(dataset_val)

#     dataset_test = DenoiseDataset(os.path.join(datadir, 'test.h5'))
#     loader_test = DataLoader(dataset_test)
        
#     train_loader, valid_loader, _ = data.build_dataset(args.dataset, args.data_path, batch_size=args.batch_size)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_psnr", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    
    my_stds = np.array([48,47,6,19])/255 
#     my_stds = np.array([10,40,23,14])/255
#     my_stds = np.array([13,7,8,10])/255 
#     my_stds = np.array([10,10,10,10])/255 

    for epoch in range(start_epoch, args.num_epochs):
        if args.resume_training:
            if epoch %10 == 0:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')
                
        # reducing learning rate after 30 epochs
        if epoch == 30:
            optimizer.param_groups[0]["lr"] /= 10
            print(" ")
            print('learning rate reduced by factor of 10')
            print(" ")
            
            
        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, inputs in enumerate(train_bar):
            model.train()

            global_step += 1
                        
            noisy_inputs_dummy = get_noisy_data_and_noise_with_same_stat_known_stds(inputs, my_stds, wavetype = 'db1',level=1)
            noisy_inputs = torch.unsqueeze(noisy_inputs_dummy[:,0,:,:],1)

            inputs = inputs.to(device)
            noisy_inputs = noisy_inputs.to(device)
            
            outputs = model(noisy_inputs)

            
            loss = F.mse_loss(outputs, inputs, reduction="sum") / (inputs.size(0) * 2)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_psnr = utils.psnr(outputs, inputs)
            train_ssim = utils.ssim(outputs, inputs)
            train_meters["train_loss"].update(loss.item())
            train_meters["train_psnr"].update(train_psnr.item())
            train_meters["train_ssim"].update(train_ssim.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
                writer.add_scalar("psnr/train", train_psnr.item(), global_step)
                writer.add_scalar("ssim/train", train_ssim.item(), global_step)
                gradients = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None], dim=0)
                writer.add_histogram("gradients", gradients, global_step)
                sys.stdout.flush()

        if epoch % args.valid_interval == 0:
            model.eval()
            for meter in valid_meters.values():
                meter.reset()

            valid_bar = utils.ProgressBar(valid_loader)
            for sample_id, sample in enumerate(valid_bar):
                with torch.no_grad():
                    

                    noisy_inputs_dummy2 = get_noisy_data_and_noise_with_same_stat_known_stds(sample, my_stds, wavetype = 'db1',level=1)
                    noisy_inputs = torch.unsqueeze(noisy_inputs_dummy2[:,0,:,:],1)
                    
                    sample = sample.to(device)
                    noisy_inputs = noisy_inputs.to(device)
                    
                    output = model(noisy_inputs)
                    
                    valid_psnr = utils.psnr(output, sample)
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_ssim = utils.ssim(output, sample)
                    valid_meters["valid_ssim"].update(valid_ssim.item())

                    if writer is not None and sample_id < 10:
                        image = torch.cat([sample, noisy_inputs, output], dim=0)
                        image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
                        writer.add_image(f"valid_samples/{sample_id}", image, global_step)

            if writer is not None:
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                sys.stdout.flush()

            logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=optimizer.param_groups[0]["lr"])))
            utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_psnr"].avg, mode="max")
        scheduler.step()

    logging.info(f"Done training! Best PSNR {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument('--datadir', type=str, default = '/storage/MRI_Stanford_2D_FSE/', help='Path to .h5 files.')

    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="dncnn", help="model architecture")

    # Add noise arguments
    parser.add_argument("--noise_mode", default="B", help="B - Blind S-one noise level")
    parser.add_argument('--noise_std', default = 25, type = float, help = 'noise level when mode is S')
    parser.add_argument('--min_noise', default = 0, type = float, help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 50, type = float, help = 'maximum noise level when mode is B')

    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=60, type=int, help="force stop training at specified epoch")
    parser.add_argument("--valid-interval", default=1, type=int, help="evaluate every N epochs")
    parser.add_argument("--save-interval", default=1, type=int, help="save a checkpoint every N steps")

    # Parse twice as model arguments are not known the first time
    parser = utils.add_logging_arguments(parser)
    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

    