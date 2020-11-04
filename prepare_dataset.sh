# remove previously generated dataset
rm -r ./data/lung_ct_dataset

# enhance images
python image_enhancement.py

# prepare tfds dataset
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=lung_ct_dataset --module_import=datasets.lung_ct_dataset --manual_dir=data/processed --data_dir=data/
