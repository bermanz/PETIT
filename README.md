# PETIT
This repository contains the WACV 2024 code submission for the paper PETIT: Physically Enhanced Thermal Images Translating GAN. (paper ID 1891).
Throughout the code/documentation, the term monochromatic and 9000nm are used interchangeably, as the 9000nm channel was the monochromatic thermal modality demonstrated in the paper.
The code was tested on Python 3.10.6, with a Nvidia RTX 4070Ti GPU.

## Setup
- Install the required packages either using either:
    - conda (recommended): `conda env create -f environment.yml`.
    - pip : `pip install -r requirements.txt`.
- Download the data and pretrained models from [here](https://drive.google.com/file/d/1NSGDsGx9hN9mQgEMI9y57v2VtyODDE7e/view?usp=sharing)
- Extract the folders in the downloaded zip to the root directory of the repository.

## Train
Run the file `train.ipynb` in the root directory of the repository.
- The notebook will save the trained models in the path `results/train/<CurrentTime>`.
- The best model in term of FID will be saved in the path `results/train/<CurrentTime>/<NetName>/best`.
- The latest model will be saved in the path `results/train/<CurrentTime>/<NetName>/latest`.

## Inference
Run the file `inference.ipynb` in the root directory of the repository.
The notebook will produce a folder named `results/transformed/<CurrentTime>` containing `.png`/`.npy` files of the generated images, depending on the user's specification.