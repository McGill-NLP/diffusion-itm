#!/bin/bash
# Download the files
wget -P datasets/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
tar -xzf datasets/pets/images.tar.gz -C datasets/pets/
rm datasets/pets/images.tar.gz

wget -P datasets/clevr https://zenodo.org/record/8096756/files/images.zip
unzip datasets/clevr/images.zip -d datasets/clevr
rm datasets/clevr/images.zip

python3 datasets/svo/download.py

wget -P datasets/imagecode https://zenodo.org/record/6518944/files/image-sets.zip
unzip datasets/imagecode/image-sets.zip -d datasets/imagecode/
