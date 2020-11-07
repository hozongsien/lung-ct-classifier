import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from model import *


def train_validate(model, train_ds, valid_ds, hyperparams, initial_epoch, num_epochs, callbacks):
    """Compiles and fit the given model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hyperparams['learning_rate']
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        initial_epoch=initial_epoch,
        epochs=num_epochs,
        callbacks=callbacks,
    )
    return model, history


def feature_extract_and_fine_tune(experiment_name, train_ds, valid_ds, model_params, base_hyperparams, fine_hyperparams):
    """Trains the model first as a feature extractor and then fine tunes the model."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs', experiment_name)
    )

    tf.keras.backend.clear_session()
    model = create_model(model_params, base_hyperparams)
    model, history = train_validate(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        hyperparams=base_hyperparams,
        initial_epoch=0,
        num_epochs=base_hyperparams['num_epochs'],
        callbacks=[tensorboard_callback]
    )

    model = create_fine_tune_model(model, fine_hyperparams)
    model, history = train_validate(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        hyperparams=fine_hyperparams,
        initial_epoch=base_hyperparams['num_epochs'],
        num_epochs=base_hyperparams['num_epochs'] +
        fine_hyperparams['num_epochs'],
        callbacks=[tensorboard_callback]
    )
    return model


def cross_validate(experiment_name, train_folds, valid_folds, model_params, base_hyperparams, fine_hyperparams, hparams):
    """Cross validates model performance with the given folds."""
    train_accs, valid_accs, train_losses, valid_losses = [], [], [], []
    for i, (train_ds, valid_ds) in enumerate(zip(train_folds, valid_folds)):
        k = i + 1
        experiment_name_fold = f'{experiment_name}: {k}-fold'
        rundir = os.path.join('logs', 'hparam_tuning', experiment_name_fold)
        print(f'# -------------------- {experiment_name_fold} -------------------- #')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(rundir)
        hparams_callback = hp.KerasCallback(rundir, hparams)

        tf.keras.backend.clear_session()
        model = create_model(model_params, base_hyperparams)
        model, history = train_validate(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            hyperparams=base_hyperparams,
            initial_epoch=0,
            num_epochs=base_hyperparams['num_epochs'],
            callbacks=[tensorboard_callback]
        )

        model = create_fine_tune_model(model, fine_hyperparams)
        model, history = train_validate(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            hyperparams=fine_hyperparams,
            initial_epoch=base_hyperparams['num_epochs'],
            num_epochs=base_hyperparams['num_epochs'] +
            fine_hyperparams['num_epochs'],
            callbacks=[tensorboard_callback, hparams_callback]
        )

        train_acc = history.history['accuracy'][-1]
        valid_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        valid_loss = history.history['val_loss'][-1]

        print(f'{experiment_name} | Train Loss: {train_loss} | Train Accuracy: {train_acc} | Validation Loss: {valid_loss} | Validation Accuracy: {valid_acc}\n')

        # models.append(model)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    avg_train_acc = np.mean(train_accs)
    avg_valid_acc = np.mean(valid_accs)
    avg_train_loss = np.mean(train_loss)
    avg_valid_loss = np.mean(valid_loss)

    print(f'Avg Train Loss: {avg_train_loss} | Avg Train Accuracy: {avg_train_acc} | Avg Validation Loss: {avg_valid_loss} | Avg Validation Accuracy: {avg_valid_acc}\n')
    return avg_train_loss, avg_train_acc, avg_valid_loss, avg_valid_acc


def ensemble_learn(experiment_name, train_ds, valid_ds, model_params, hyperparams):
    """Trains an ensemble of pretrained models."""
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('logs', experiment_name)
    )

    tf.keras.backend.clear_session()
    model = create_ensemble_model(model_params, hyperparams)
    model, history = train_validate(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        hyperparams=hyperparams,
        initial_epoch=0,
        num_epochs=hyperparams['num_epochs'],
        callbacks=[tensorboard_callback]
    )
    return model


def evaluate(model, test_ds):
    """Predicts labels on the given test dataset."""
    predictions = model.predict(test_ds)
    predicted_indices = tf.argmax(predictions, 1)
    predicted_labels = predicted_indices.numpy()
    return predicted_labels
