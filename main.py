import tensorflow_datasets as tfds
from data_preprocessing import *
from model import *
from train import *
from utils import *


def train_model():
    data_folder = 'data'
    raw_folder = 'raw'
    processed_folder = 'processed'
    dataset = 'lung_ct_dataset'
    train_folder = 'train'
    test_folder = 'test'
    train_label = 'train_label.csv'

    SEED = 0

    experiment_name = 'MobileNetV2'
    result_save_path = experiment_name + '_' + 'submission.csv'
    model_params = {
        'model_type': 'mobile',
        'head_type': 'standard',
        'image_shape': (512, 512, 3),
        'num_classes': 3,
    }
    base_hyperparams = {
        'train_batch_size': 64,
        'valid_batch_size': 64,
        'test_batch_size': 64,
        'num_epochs': 1,
        'learning_rate': 1e-4,
        'dropout': 0.2
    }
    fine_hyperparams = {
        'num_epochs': 1,
        'learning_rate': 1e-5,
        'fine_tune_at': 100,
    }

    tf.random.set_seed(SEED)
    gpu_setup()
    mixed_precision_setup()

    train_folds = tfds.load(
        name=dataset,
        split=[f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 20)],
        download=False,
        shuffle_files=False,
        as_supervised=True,
        data_dir=data_folder
    )
    valid_folds = tfds.load(
        name=dataset,
        split=[f'train[{k}%:{k+10}%]' for k in range(0, 100, 20)],
        download=False,
        shuffle_files=False,
        as_supervised=True,
        data_dir=data_folder
    )
    test_ds_raw, test_info_raw = tfds.load(
        name=dataset,
        split='test',
        download=False,
        shuffle_files=False,
        as_supervised=False,
        with_info=True,
        data_dir=data_folder
    )

    train_ds = preprocess(
        train_folds[0], model_params, batch_size=base_hyperparams['train_batch_size'], ds_type='train')
    valid_ds = preprocess(
        valid_folds[0], model_params, batch_size=base_hyperparams['valid_batch_size'], ds_type='valid')
    test_ds = preprocess(test_ds_raw, model_params,
                         batch_size=base_hyperparams['test_batch_size'], ds_type='test')

    train_ds = train_ds.concatenate(valid_ds)  # train on entire dataset

    model = feature_extract_and_fine_tune(
        experiment_name, train_ds, valid_ds, model_params, base_hyperparams, fine_hyperparams)
    predicted_labels = evaluate(model, test_ds)
    img_ids = tfds.as_dataframe(test_ds_raw, test_info_raw)
    save_results(img_ids, predicted_labels, result_save_path)


def train_ensemble():
    pass


def main():
    train_model()


if __name__ == "__main__":
    main()
