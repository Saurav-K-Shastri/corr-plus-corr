import torch
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
import time

def wave_forward_list(image, xfm):
    """
    return wavelet in generic [Yl,Yh] list
    image size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
#     device = image.device
#     xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(image)
    return [Yl, Yh]


def wave_inverse_list(wave_list, ifm):
    """
    return image of size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
    Yl, Yh = wave_list
#     device = Yl.device
#     ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
    my_image = ifm((Yl, Yh))
    

    return my_image


def wave_forward_mat(image, xfm):
    """
    retruns wavelet matrix of same size as image
    image size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    
    """
#     device = image.device
#     xfm = DWTForward(J=level, mode='symmetric', wave=wavelet).to(device)  # Accepts all wave types available to PyWavelets
    
#     t = time.time()
    Yl, Yh = xfm(image)
    


    my_wave_mat = get_wave_mat([Yl,Yh])

#     elapsed = time.time() - t
    
#     print('time take: ')
#     print(elapsed)
#     print(my_wave_mat.shape)

    return my_wave_mat


def wave_inverse_mat(wave_mat, ifm, level):
    """
    retruns image of size : [batch, channel, H, W ] channel is 2 to represent complex numbers 
    """
    Yl, Yh = wave_mat2list(wave_mat,level)
#     device = Yl.device
#     ifm = DWTInverse(mode='symmetric', wave=wavelet).to(device)
    my_image = ifm((Yl, Yh))
    return my_image



def get_wave_mat(wave_list):
    Yl,Yh = wave_list
    level = len(Yh)
    my_mat = Yl.clone()

    for i in range(level):
        index = level - 1 - i
        my_mat = torch.cat((torch.cat((my_mat,Yh[index][:,:,0,:,:].clone()),dim = 2),torch.cat((Yh[index][:,:,1,:,:].clone(),Yh[index][:,:,2,:,:].clone()),dim = 2)),dim = 3)
    return my_mat


def wave_mat2list(wave_mat,level = 4):
    
    assert wave_mat.shape[2] == wave_mat.shape[3], "last two dimentions should be same"
    
    N = wave_mat.shape[2]
    srt_LH_row = int(N/2)
    end_LH_row = N
    srt_LH_col = 0
    end_LH_col = int(N/2)
    
    Yh = []
    for i in range(level):
        Yh.append(torch.stack((wave_mat[:,:,srt_LH_row:end_LH_row,srt_LH_col:end_LH_col].clone(),wave_mat[:,:,srt_LH_col:end_LH_col,srt_LH_row:end_LH_row].clone(), wave_mat[:,:,srt_LH_row:end_LH_row,srt_LH_row:end_LH_row].clone()),dim = 2))
        end_LH_row = srt_LH_row
        srt_LH_row = int(srt_LH_row/2)
        srt_LH_col = 0
        end_LH_col = int(end_LH_col/2)

    Yl = wave_mat[:,:,0:end_LH_row,0:end_LH_row].clone()

    return [Yl,Yh]


def add_noise_subbandwise_list_batch(wave_list,stds):
    
    """ add noise of different noise levels to each subband """
    Yl,Yh = wave_list
    device = Yl.device
    level = len(Yh)
    
    is_complex = (Yh[0].shape[1] == 2)
    
    Yl_new,Yh_new = wave_mat2list(get_wave_mat(wave_list).clone())
    
    Yl_new = Yl_new + generate_noise_mat_batch(Yl.shape,stds[:,0],is_complex,device)
    
    count = 1
    for i in range(level):
        index = level-1-i
        for j in range(3):
            Yh_new[index][:,:,j,:,:] = Yh_new[index][:,:,j,:,:] + generate_noise_mat_batch(Yh_new[index][:,:,j,:,:].shape,stds[:,count],is_complex,device) 
            count = count+1
    
    return [Yl_new,Yh_new]

def generate_noise_mat_batch(my_shape,my_std,is_complex,device):
    
    if is_complex:
        my_std_new = my_std.clone()/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = my_std.clone()
    
    my_noise_mat = (my_std_new*((torch.randn(my_shape, device = device)).permute(1,2,3,0))).permute(3,0,1,2)
    
    return my_noise_mat