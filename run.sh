#!/bin/bash

# prepare dataset
bash prepare_dataset.sh $1 $2

# train and evaluate
python src/main.py
