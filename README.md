# Introduction

This proposed model aims to classify lung computerised tomography (CT) scans into 3 anoynmised catergories using an ensemble of 3 models MobileNetV2, Xception and ResNet152V2. This approach achieved 94.52% accuracy on an unseen test set.

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
2. Download the dataset and unzip the files to your desired location.

   ```
   unzip -a data.zip
   ```

   > NOTE: Verify directory structure.
   >
   > ```
   > ├── train_image/
   > │    └── train_image/
   > │        └── .png
   > ├── test_image/
   > │    └── test_image/
   > │        └── .png
   > ├── train_label.csv
   > └── ...
   > ```

3. Assign the path of the unzipped dataset chosen in step 2 to SRC_PATH in the `prepare_dataset.sh` script.

   ```
   vim prepare_dataset.sh
   ```

4. Execute script to train models and make predictions on test dataset.
   ```
   bash run.sh
   ```

> NOTE: The Lung CT scan dataset is a private dataset.

# Logging

This code comes with tensorboard support to track the loss curves and evaluation metrics of the proposed model. Launch tensorboard using this command below.

```
tensorboard --logdir=logs
```

# Visulisation and Tuning

This code comes with a companion jupyter notebook `tune.ipynb` to visualise the output of the data preprocessing and to select the best hyperparameters to train each model.

1. Execute the same steps 1-3 from [Quick Start](#quick-start) above.

2. Launch jupyter.
   ```
   jupyter lab
   ```

# Approach Summary

1. The images are first enhanced using a technique called Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve contrast of the CT scans.
2. Data augmentation techniques, random flips and rotations are used to improve diversity of the small dataset
3. Each of the 3 models (MobileNetV2, Xception and ResNet152V2) are pretrained on the imagenet dataset and used as feature extractors. The models are then fine tuned by unfreezing some layers.
4. K-fold cross validation is used to select the best hyperparameters for each model which are then used to train a integrated stacked ensemble model.
