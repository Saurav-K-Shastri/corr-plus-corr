# corr+corr: A DNN denoiser for correlated noise

Authors: Saurav K. Shastri, Rizwan Ahmad, Christopher A. Metzler, and Philip Schniter

The corr+corr denoiser is described in Section III-B of our paper: "Denoising Generalized Expectation-Consistent Approximation for MRI Image Recovery"

arXiv Link: https://arxiv.org/pdf/2206.05049.pdf

The 'Denoiser_Experiments' folder contains the code files required for generating the results shown in Section IV-A of the paper.

To make sure that the versions of the packages match those we tested the code with, we recommend creating a new virtual environment and installing packages using `conda` with the following command.

```bash
conda env create -f environment.yml
```