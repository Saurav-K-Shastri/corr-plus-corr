Run 'main_data_generation.py' to generate the .h5 files that will be used in training the denoisers.\
Alternatively, you can download the pregenerated files from here: https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/shastri_19_buckeyemail_osu_edu/Em4IeIm6dwtBrKrzYxzn7gAB8uPlTSeLRcPE2AvWRMr4sA?e=QVTlPi

To train various denoisers, use the following files:
* wite DnCNN: 'train_BF_CNN_image_denoiser.py'
* Metzler’s DnCNN: 'train_CNN_Metzler_coloured_image_denoiser.py'
* corr+corr DnCNN: 'train_BF_CNN_corr_p_corr_image_denoiser.py'
* genie DnCNN: 'train_BF_CNN_image_denoiser_correlated_noise.py'