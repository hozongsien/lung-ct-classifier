import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def gpu_setup():
    """Setup the available GPUs to be used for training."""
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        # Specify GPU to restrict usage below
        tf.config.set_visible_devices(physical_devices, 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        # Logical device was not created for first GPU
        assert len(logical_devices) == len(physical_devices) - 1
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def mixed_precision_setup():
    """Setup support for mixed precision training."""
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


def plot(acc, val_acc, loss, val_loss, initial_epochs=0):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1.0])
    plt.title('Training and Validation Accuracy')
    if initial_epochs != 0:
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    if initial_epochs != 0:
        plt.plot([initial_epochs-1, initial_epochs-1],
                 plt.ylim(), label='Start Fine Tuning')
    plt.show()


def save_results(image_ids, predicted_labels, save_path):
    """Saves the predicted labels to csv."""
    print('Saving predictions\n')
    results = image_ids.drop('image', axis=1)
    results.columns = ['ID', 'Label']
    results['Label'] = predicted_labels
    results = results.sort_values('ID').reset_index(drop=True)
    results.to_csv(save_path, index=False)


def save_model(model, experiment_name):
    """Saves the given model."""
    print('Saving model\n')
    filename = os.path.join('models', experiment_name, '.h5')
    model.save(filename)
