# Introduction

This proposed model aims to classify lung CT scans into 3 anoynmised catergories using an ensemble of 3 models MobileNetV2, Xception and ResNet152V2. This approach achieved 94.52% accuracy on an unseen test set.

# Prerequisite

- python >= 3.7.9
- virtualenv

# Quick Start

The instructions below trains and evaluates the proposed model on the lung CT dataset.

1. Setup virtual environment.

   ```
   virtualenv env
   source env/bin/activate
   pip3 install -r requirements.txt
   ```

2. Execute script to train models and evaluate on test dataset.
   ```
   bash run.sh train_data test_data
   ```

# Approach Summary

1. The images are first enhanced using a technique called Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve contrast of the CT scans.
2. Data augmentation techniques, random flips and rotations are used to improve diversity of the small dataset
3. Each of the 3 models (MobileNetV2, Xception and ResNet152V2) are pretrained on the imagenet dataset and used as feature extractors. The models are then fine tuned by unfreezing some layers.
4. K-fold cross validation is used to select the best hyperparameters for each model which are then used to train a soft voting ensemble model.
