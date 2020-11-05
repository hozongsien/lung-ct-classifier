#!/bin/bash

SRC_PATH=nus-cs5242 # replace with source data path here
DST_PATH=data/processed


# clear previous
rm -r ./data/lung_ct_dataset

# create folders to store processed images
mkdir data $DST_PATH $DST_PATH/train $DST_PATH/test

# enhance images
python src/image_enhancement.py $SRC_PATH $DST_PATH

# create tfds dataset
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=lung_ct_dataset --module_import=src.datasets.lung_ct_dataset --manual_dir=$DST_PATH --data_dir=data/
