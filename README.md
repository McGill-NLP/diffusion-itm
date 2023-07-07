# DiffusionITM
Code and data setup for our paper [Are Diffusion Models Vision-and-language Reasoners?](https://arxiv.org/abs/2305.16397)

<img src="images/mainfig.jpeg" width="500" height="500">

Work-in-progress. Code and data will be fully released the next weeks (end of June).

## Setup
IMPORTANT: Clone the repository with the sumbodules option:
```
git clone --recurse-submodules git@github.com:McGill-NLP/diffusion-itm.git
```

git clone --recurse-submodules ...

Install torch & torchvision.
Navigate to diffusers and run setup.py install

## Dataset Setup

Run `setup.sh` to download images for several of the datasets (CLEVR, SVO, ImageCoDe, Pets).
For the rest, there are some small manual steps:

### Flickr30K

Download the images from [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and save them under datasets: `datasets/flickr30k/images`.

<!-- ### ARO

Nothing to do since the ARO repository will download VG and COCO by itself. -->

<!-- ### Pets
Images: https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz

### CLEVR

```
wget https://zenodo.org/record/8096756/files/images.zip
```

### SVO

Run datasets/svo/download.py

### ImageCoDe

wget https://zenodo.org/record/6518944/files/image-sets.zip -->

### Winoground

Fill in AUTH_TOKEN in line 259

### Bias

We will have instructions for these datasets soon.

## Zero-shot Image-Text-Matching

python3 diffusion_itm.py --task flickr30k

## Hard Negative Finetuning

