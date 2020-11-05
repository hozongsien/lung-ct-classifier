#!/bin/bash

SRC_TRAIN_DATA_PATH=$1
SRC_TEST_DATA_PATH=$2 
DST_DATA_PATH=data/processed


# clear previous
rm -r ./data/lung_ct_dataset

# create folders to store processed images
mkdir data $DST_DATA_PATH $DST_DATA_PATH/train $DST_DATA_PATH/test

# enhance images
python src/image_enhancement.py $SRC_TRAIN_DATA_PATH $SRC_TEST_DATA_PATH $DST_DATA_PATH

# create tfds dataset
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=lung_ct_dataset --module_import=src.datasets.lung_ct_dataset --manual_dir=$DST_DATA_PATH --data_dir=data/
