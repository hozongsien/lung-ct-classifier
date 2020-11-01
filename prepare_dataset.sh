# remove previously generated dataset
rm -r ./data/mri_dataset

# pre pre-process images
python preprocess.py

# prepare tfds dataset
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=mri_dataset --module_import=datasets.mri_dataset --manual_dir=data/processed --data_dir=data/
