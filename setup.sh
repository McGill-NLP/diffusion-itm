#!/bin/bash
# Download the files
wget -P data/pets https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
tar -xzf data/pets/images.tar.gz -C data/pets/
rm data/pets/images.tar.gz

wget -P data/clevr https://zenodo.org/record/8096756/files/images.zip
unzip data/clevr/images.zip -d data/clevr
rm data/clevr/images.zip

wget -P data/imagecode https://zenodo.org/record/6518944/files/image-sets.zip
unzip data/imagecode/image-sets.zip -d data/imagecode/

python3 data/svo/download.py