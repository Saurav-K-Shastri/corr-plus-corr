import torch

import utils.wave_torch_transform_for_denoiser as wutils

def get_noise(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    noise = torch.randn_like(data);
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (max_noise - min_noise) * torch.rand(n) + min_noise;
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
    else:
        noise.data = noise.data * noise_std;
    return noise


def get_noisy_data_with_SD_map_complex(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    

    noise = torch.randn_like(data);
    
    result_data = torch.zeros(data.shape[0],3,data.shape[2],data.shape[3]);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = noise_tensor_array[i]*torch.ones(data[i,0,:,:].shape)
    else:
        noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
            result_data[i,2,:,:] = (1/torch.sqrt(torch.tensor(2)))*noise_std*torch.ones(data[i,0,:,:].shape)
        
    return result_data


def get_complex_noisy_data_and_noise_realization(data, min_noise, max_noise, xfm, ifm, level):
    
    num_of_sbs = 3*level + 1
    
    device = data.device
    
    result_data = torch.zeros((data.shape[0],3,data.shape[2],data.shape[3]),device = device);
    
    stds = ((max_noise - min_noise) * torch.rand((data.shape[0], num_of_sbs), device = device) + min_noise);
    
    result_data[:,0:2,:,:] = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list_batch(wutils.wave_forward_list(data.clone(), xfm),stds),ifm) # this handles complex noise
    
    noise_realization = wutils.wave_inverse_list(wutils.add_noise_subbandwise_list_batch(wutils.wave_forward_list(torch.zeros_like(data), xfm),stds),ifm) # this handles complex noise
    
    result_data[:,2,:,:] = noise_realization[:,0,:,:]
    
    return result_data


