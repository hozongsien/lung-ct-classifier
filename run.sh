#!/bin/bash

# prepare dataset
bash prepare_dataset.sh

# train and evaluate
mkdir models
python src/main.py
