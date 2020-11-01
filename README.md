# Quick Start

1. Setup environment.
   ```
   virtualenv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
2. Create folders for storing data.
   ```
   mkdir data data/raw data/processed data/processed/train data/processed/test
   ```
3. Download data from kaggle and unzip files.
   ```
   unzip -a nus-cs5242.zip -d data/raw
   ```
4. Preprocess images and create dataset.
   ```
   bash prepare_dataset.sh
   ```
5. Train model using jupyter notebook.
   ```
   jupyter lab
   ```
