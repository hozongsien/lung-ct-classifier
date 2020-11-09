import tensorflow_datasets as tfds
from data_preprocessing import *
from model import *
from train import *
from utils import *


def train_model(experiment_name, model_params, base_hyperparams, fine_hyperparams):
    data_folder = 'data'
    dataset = 'lung_ct_dataset'

    SEED = 0

    result_save_path = experiment_name + '_' + 'submission.csv'

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
    img_ids = tfds.as_dataframe(test_ds_raw, test_info_raw)

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
    save_results(img_ids, predicted_labels, result_save_path)
    save_model(model, experiment_name)


def train_ensemble(experiment_name, model_params, base_hyperparams):
    data_folder = 'data'
    dataset = 'lung_ct_dataset'

    SEED = 0

    result_save_path = experiment_name + '_' + 'submission.csv'

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
    img_ids = tfds.as_dataframe(test_ds_raw, test_info_raw)

    train_ds = preprocess(
        train_folds[0], model_params, batch_size=base_hyperparams['train_batch_size'], ds_type='train')
    valid_ds = preprocess(
        valid_folds[0], model_params, batch_size=base_hyperparams['valid_batch_size'], ds_type='valid')
    test_ds = preprocess(test_ds_raw, model_params,
                         batch_size=base_hyperparams['test_batch_size'], ds_type='test')

    train_ds = train_ds.concatenate(valid_ds)  # train on entire dataset

    train_ds = preprocess_ensemble(train_ds, model_params)
    valid_ds = preprocess_ensemble(valid_ds, model_params)
    test_ds = preprocess_ensemble(test_ds, model_params)

    ensemble_model = ensemble_learn(
        experiment_name, train_ds, valid_ds, model_params, base_hyperparams)
    predicted_labels = evaluate(ensemble_model, test_ds)
    save_results(img_ids, predicted_labels, result_save_path)


def main():
    model_configs = {
        'Xception': {
            'model_params': {
                'model_type': 'Xception',
                'head_type': 'standard',
                'image_shape': (512, 512, 3),
                'num_classes': 3,
            },
            'base_hyperparams': {
                'train_batch_size': 64,
                'valid_batch_size': 64,
                'test_batch_size': 64,
                'num_epochs': 300,
                'learning_rate': 1e-4,
                'dropout': 0.5
            },
            'fine_hyperparams': {
                'num_epochs': 150,
                'learning_rate': 1e-5,
                'fine_tune_at': 80,
            }
        },
        'MobileNetV2': {
            'model_params': {
                'model_type': 'MobileNetV2',
                'head_type': 'standard',
                'image_shape': (512, 512, 3),
                'num_classes': 3,
            },
            'base_hyperparams': {
                'train_batch_size': 64,
                'valid_batch_size': 64,
                'test_batch_size': 64,
                'num_epochs': 150,
                'learning_rate': 1e-4,
                'dropout': 0.2
            },
            'fine_hyperparams': {
                'num_epochs': 50,
                'learning_rate': 1e-5,
                'fine_tune_at': 100,
            }
        },
        'ResNet152V2': {
            'model_params': {
                'model_type': 'ResNet152V2',
                'head_type': 'standard',
                'image_shape': (512, 512, 3),
                'num_classes': 3,
            },
            'base_hyperparams': {
                'train_batch_size': 64,
                'valid_batch_size': 64,
                'test_batch_size': 64,
                'num_epochs': 200,
                'learning_rate': 5e-4,
                'dropout': 0.25
            },
            'fine_hyperparams': {
                'num_epochs': 100,
                'learning_rate': 1e-5,
                'fine_tune_at': 544,
            }
        },
    }

    ensemble_config = {
        'model_params': {
            'model_names': model_configs.keys(),
            'head_type': 'ensemble',
            'image_shape': (512, 512, 3),
            'num_classes': 3,
        },
        'base_hyperparams': {
            'train_batch_size': 64,
            'valid_batch_size': 64,
            'test_batch_size': 64,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'hidden_units': 16
        }
    }

    for model, config in model_configs.items():
        train_model(model, **config)

    train_ensemble('Ensemble', **ensemble_config)


if __name__ == "__main__":
    main()
