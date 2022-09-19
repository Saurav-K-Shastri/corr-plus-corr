# corr+corr: A DNN denoiser for correlated noise

Authors: Saurav K. Shastri, Rizwan Ahmad, Christopher A. Metzler, and Philip Schniter

The corr+corr denoiser is described in Section III-B of our paper: "Denoising Generalized Expectation-Consistent Approximation for MRI Image Recovery"

The paper has been accepted for publication as a Special Issue on Deep Learning Methods for Inverse Problems paper in the IEEE Journal on Selected Areas in Information Theory.

IEEE link: https://ieeexplore.ieee.org/document/9893363

arXiv link (includes supplementary material): https://arxiv.org/pdf/2206.05049.pdf

The 'Denoiser_Experiments' folder contains the code files required for generating the results shown in Section IV-A of the paper.

The 'fastMRI_corr_plus_corr' folder contains the code files required to train the complex image denoisers on fastMRI datasets.

To make sure that the versions of the packages match those we tested the code with, we recommend creating a new virtual environment and installing packages using `conda` with the following command.

```bash
conda env create -f environment.yml
```
