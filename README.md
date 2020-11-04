# Introduction

This proposed model aims to classify lung CT scans into 3 anoynmised catergories using an ensemble of 3 models MobileNetV2, Xception and ResNet152V2. This approach achieved 94.52% accuracy on an unseen test set.

# Prerequisite

- python 3.7.9

# Quick Start

The instructions below trains and evaluates the proposed model on the lung CT dataset.

1. Setup virtual environment.
   ```lang-bash
   virtualenv env
   source env/bin/activate
   pip3 install -r requirements.txt
   ```
2. Download the dataset and unzip the files to your desired location.
   ```lang-bash
   unzip -a nus-cs5242.zip
   ```
3. Assign the path of the unzipped dataset in step 2 to SRC_PATH in the `prepare_dataset.sh` script.

   ```lang-bash
   vim prepare_dataset.sh
   ```

4. Execute script to train models and evaluate on test dataset.
   ```lang-bash
   bash run.sh
   ```

# Logging

This code comes with tensorboard support to track the loss curves and evaluation metrics of the proposed model. Launch tensorboard using this command below.

```lang-bash
tensorboard --logdir=logs
```

# Visulisation and Tuning

This code comes with a companion jupyter notebook `tune.ipynb` to visualise the output of the data preprocessing and to select the best hyperparameters to train each model.

1. Execute the same steps 1-3 from [Quick Start](#quick-start) above.

2. Launch jupyter.
   ```lang-bash
   jupyter lab
   ```

# Approach Summary

1. The images are first enhanced using a technique called Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve contrast of the CT scans.
2. Data augmentation techniques, random flips and rotations are used to improve diversity of the small dataset
3. Each of the 3 models (MobileNetV2, Xception and ResNet152V2) are pretrained on the imagenet dataset and used as feature extractors. The models are then fine tuned by unfreezing some layers.
4. K-fold cross validation is used to select the best hyperparameters for each model which are then used to train a soft voting ensemble model.
