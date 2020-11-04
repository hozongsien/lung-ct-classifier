#!/bin/bash

SRC_PATH=data/raw
DST_PATH=data/processed


# clear previous
rm -r ./data/lung_ct_dataset

# create folders
mkdir data $SRC_PATH $DST_PATH $DST_PATH/train $DST_PATH/test
mkdir models

# enhance images
python image_enhancement.py # TODO: $SRC_PATH $DST_PATH

# create tfds dataset
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=lung_ct_dataset --module_import=datasets.lung_ct_dataset --manual_dir=$DST_PATH --data_dir=data/

# train and evaluate
python main.py
