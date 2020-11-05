#!/bin/bash
scriptdir="$(dirname "$0")"
cd "$scriptdir"

# prepare dataset
bash prepare_dataset.sh ../$1 ../$2

# train and evaluate
python src/main.py
