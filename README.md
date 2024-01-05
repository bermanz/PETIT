<!-- TODO: add citation to both this markdown and website once bibtex is ready -->
# PETIT-GAN
This repository contains the source code for the methods and results introced in the paper *PETIT: Physically Enhanced Thermal Images Translating GAN*, which was published in [WACV 2024](https://wacv2024.thecvf.com/)

Throughout the code/documentation, the term monochromatic and 9000nm are used interchangeably, as the 9000nm channel was the monochromatic thermal modality demonstrated in the paper.

For more details about the research, please checkout our [project's website](https://bermanz.github.io/PETIT/)

![PETIT](docs/figs/methods/results_comp_ps.png)  


## Setup
- Install the required packages either using either:
    - conda (recommended): `conda env create -f environment.yml`.
    - pip : `pip install -r requirements.txt`.
- Run the `download.py` script to download and unpack all the data and pretrained models. **_NOTE:_**  The data used here is a fraction of the entire dataset (both in terms of amounts of images and modalities). For the full data set, click [here](https://agropticslab.volcani.institute/PETIT-GAN/data).
- Extract the folders in the downloaded zip to the root directory of the repository.
- After completing the setup, the project's root should have the following structure:

```
├── data
├── docs
├── models
├── src
├── .gitignore
├── download.py
├── environment.yml
├── inference.ipynb
├── requirements.txt
├── train.ipynb
```

## Train
Run the file `train.ipynb` in the root directory of the repository.
- The notebook will save the trained models in the path `results/train/<CurrentTime>`.
- The best model in term of FID will be saved in the path `results/train/<CurrentTime>/<NetName>/best`.
- The latest model will be saved in the path `results/train/<CurrentTime>/<NetName>/latest`.

## Inference
Run the file `inference.ipynb` in the root directory of the repository.
The notebook will produce a folder named `results/transformed/<CurrentTime>` containing `.png`/`.npy` files of the generated images, depending on the user's specification in the notebook.

## Citation
If you use PETIT-GAN's code/paper for your research, please cite using the following BibTex:

```
@InProceedings{Berman_2024_WACV,
  author    = {Berman, Omri and Oz, Navot and Mendlovic, David and Sochen, Nir and Cohen, Yafit and Klapp, Iftach},
  title     = {PETIT-GAN: Physically Enhanced Thermal Image-Translating Generative Adversarial Network},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month     = {January},
  year      = {2024},
  pages     = {1618-1627}
}
```
