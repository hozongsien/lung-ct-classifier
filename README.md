# Quick Start

The instruction below trains and evaluates the proposed model on the lung CT dataset.

1. Setup virtual environment.
   ```
   virtualenv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
2. Download the dataset and unzip the files to your desired location.
   ```
   unzip -a nus-cs5242.zip
   ```
3. Assign the path of the unzipped dataset in step 2 to SRC_PATH in the run.sh script.

   ```
   vim run.sh
   ```

4. Execute script to train and evaluate on test dataset.
   ```
   bash run.sh
   ```

# Visulisation and Tuning

This code comes with a companion jupyter notebook `tune.ipynb` to visualise the output of the data preprocessing and to select the best hyperparameters to train each model. Launch the jupyter notebook using this command below.

```
jupyter lab
```

# Logging

This code comes with tensorboard support to track the loss curves and evaluation metrics of the proposed model. Launch tensorboard using this command below.

```
tensorboard --logdir=logs
```
