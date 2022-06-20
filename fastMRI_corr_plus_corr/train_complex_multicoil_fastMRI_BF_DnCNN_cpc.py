## Author: Saurav
## I used the following python command to run the file: PYTHONPATH=. python 'train_complex_multicoil_fastMRI_BF_DnCNN_cpc.py' --output-dir '/storage/fastMRI/trained_models/BF_DnCNN/complex_multicoil_fastMRI_knee_BF_DnCNN_cpc' --min_noise 0 --max_noise 10 --batch-size 128 --lr 1e-3

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

from utils.noise_model import get_noisy_data_with_SD_map_complex

from utils.noise_model import get_complex_noisy_data_and_noise_realization

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)

from data_fastMRI import transforms_new

from utils.dataset import DenoiseDataset

from torch.utils.data import Sampler

class MyRandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples
    
    
def main(args):
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: {}'.format(device))
    
    wavelet = args.wavelet
    level = args.level

    xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)

    utils.setup_experiment(args)
    utils.init_logging(args)

    # Build data loaders, a model and an optimizer
    model = models.build_model(args)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12, 14, 16, 18, 19], gamma=0.5)
    
    logging.info(f"Built a model consisting of {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.resume_training:
        state_dict = utils.load_checkpoint(args, model, optimizer, scheduler)
        global_step = state_dict['last_step']
        start_epoch = int(state_dict['last_step']/(552576/state_dict['args'].batch_size))+1 # 552576 -> replace with num_of_batches *batch_size.
        optimizer.param_groups[0]["lr"] = 1e-03
    else:
        global_step = -1
        start_epoch = 0
    
    print('Parsing training arguments...')
    datadir = args.datadir
    print('Parameters:')
    print_arguments(args)
    print()
    
    print('Setting up training...')
    
    dataset_train = DenoiseDataset(os.path.join(datadir, 'train.h5'))
    
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers= 8)

    dataset_val = DenoiseDataset(os.path.join(datadir, 'valid.h5'))

    valid_loader = DataLoader(dataset_val, num_workers=8)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss", "train_ssim"])}
    valid_meters = {name: utils.AverageMeter() for name in (["valid_loss", "valid_psnr", "valid_ssim"])}
    writer = SummaryWriter(log_dir=args.experiment_dir) if not args.no_visual else None

    for epoch in range(start_epoch, args.num_epochs):
        
        if args.resume_training:
            if epoch == 8 or epoch == 12 or epoch == 14 or epoch == 16 or epoch == 18 or epoch == 19:
                optimizer.param_groups[0]["lr"] /= 2
                print('learning rate reduced by factor of 2')
            
        train_bar = utils.ProgressBar(train_loader, epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, inputs in enumerate(train_bar):
            model.train()

            global_step += 1
            inputs = inputs.to(device)
            
            noisy_inputs_with_noise_realization = get_complex_noisy_data_and_noise_realization(inputs, args.min_noise/255., args.max_noise/255., xfm, ifm,level)
                        
            outputs = model(noisy_inputs_with_noise_realization)
            
            loss = F.mse_loss(outputs, inputs, reduction="sum") / (inputs.size(0) * 2)
                
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_ssim = utils.ssim_fastMRI(transforms_new.complex_abs(inputs.permute(0,2,3,1)), transforms_new.complex_abs(outputs.permute(0,2,3,1)))
        
            train_meters["train_loss"].update(loss.item())
            train_meters["train_ssim"].update(train_ssim.item())
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            if writer is not None and global_step % args.log_interval == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("loss/train", loss.item(), global_step)
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
                    sample = sample.to(device)

                    noisy_sample_with_noise_realization = get_complex_noisy_data_and_noise_realization(sample, args.min_noise/255., args.max_noise/255., xfm, ifm,level)
                    noisy_sample = noisy_sample_with_noise_realization[:,0:2,:,:] # Does not provide the SD values
                    
                    output = model(noisy_sample_with_noise_realization)
                    
                    
                    valid_loss = F.mse_loss(output, sample, reduction="sum") / (sample.size(0) * 2)
                    
                    valid_meters["valid_loss"].update(valid_loss.item())
                    valid_psnr = utils.psnr_fastMRI(transforms_new.complex_abs(sample.permute(0,2,3,1)),transforms_new.complex_abs(output.permute(0,2,3,1))) 
                    valid_meters["valid_psnr"].update(valid_psnr.item())
                    valid_ssim = utils.ssim_fastMRI(transforms_new.complex_abs(sample.permute(0,2,3,1)), transforms_new.complex_abs(output.permute(0,2,3,1)))
                    valid_meters["valid_ssim"].update(valid_ssim.item())


                    if writer is not None and sample_id < 10:
                        image = torch.cat([sample, noisy_sample[:,0:2,:,:], output], dim=0)
                        image = torchvision.utils.make_grid(image.clamp(0, 1), nrow=3, normalize=False)
                        writer.add_image(f"valid_samples/{sample_id}", image, global_step)

            if writer is not None:
                writer.add_scalar("loss/valid", valid_meters['valid_loss'].avg, global_step)
                writer.add_scalar("psnr/valid", valid_meters['valid_psnr'].avg, global_step)
                writer.add_scalar("ssim/valid", valid_meters['valid_ssim'].avg, global_step)
                sys.stdout.flush()

            logging.info(train_bar.print(dict(**train_meters, **valid_meters, lr=optimizer.param_groups[0]["lr"])))
            if args.data_parallel:
                utils.save_checkpoint_data_parallel(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
            else:
                utils.save_checkpoint(args, global_step, model, optimizer, score=valid_meters["valid_loss"].avg, mode="min")
        scheduler.step()
        
    if args.data_parallel:
        logging.info(f"Done training! Best loss {utils.save_checkpoint_data_parallel.best_score:.3f} obtained after step {utils.save_checkpoint_data_parallel.best_step}.")
    else:
        logging.info(f"Done training! Best loss {utils.save_checkpoint.best_score:.3f} obtained after step {utils.save_checkpoint.best_step}.")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument('--datadir', type=str, default = '/storage/fastMRI/data/multicoil_knee/pre_processed/', help='Path to images or .h5 files.') # Use this for knee dataset
#     parser.add_argument('--datadir', type=str, default = '/storage/fastMRI/data/multicoil_brain/pre_processed/', help='Path to images or .h5 files.') # Use this for brain dataset

    parser.add_argument("--batch-size", default=128, type=int, help="train batch size")

    # Add model arguments
    parser.add_argument("--model", default="dncnn_cpc", help="model architecture")

    # Add noise arguments
    parser.add_argument('--min_noise', default = 0, type = int, help = 'minimum noise level when mode is B')
    parser.add_argument('--max_noise', default = 10, type = int, help = 'maximum noise level when mode is B')
    
    # Add wavelet arguments
    parser.add_argument("--wavelet", type=str, default= 'haar', help="enter wavelet to use")
    parser.add_argument('--level', default = 4, type = int, help = 'wavelet level to use')


    # Add optimization arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num-epochs", default=20, type=int, help="force stop training at specified epoch")
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
