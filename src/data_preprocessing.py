import tensorflow as tf


def preprocess(ds, model_params, batch_size, ds_type='train'):
    """Preprocess tfds dataset."""
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    resize_and_rescale = create_resize_and_rescale_layer(model_params)
    data_augmentation = create_augmentation_layer()

    if ds_type == 'test':
        ds = ds.map(lambda ds: (ds['image'], ds['label']))

    # disable to visualise train/ valid images
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if ds_type == 'train':
        ds = ds.map(lambda x, y: (data_augmentation(x), y),
                    num_parallel_calls=AUTOTUNE)

    if ds_type == 'train' or ds_type == 'valid':
        ds = ds.map(lambda x, y: (x, tf.one_hot(
            y, depth=model_params['num_classes'])))

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def preprocess_ensemble(ds, model_params):
    """Preprocess tfds dataset for ensemble model with multiple input heads."""
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.map(lambda x, y: (
        {'ensemble_0_input_2': x, 'ensemble_1_input_2': x, 'ensemble_2_input_2': x}, y))
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def create_resize_and_rescale_layer(model_params):
    """Creates a sequential layer that resizes and rescale input images."""
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(
            model_params['image_shape'][0],
            model_params['image_shape'][1],
            interpolation='bilinear'
        ),
        tf.keras.layers.experimental.preprocessing.Rescaling(
            1./127.5, offset=-1)
    ], name='resize_and_rescale')
    return resize_and_rescale


def create_augmentation_layer():
    """Creates a sequential layer that applies data augmentation on input images."""
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ], name='data_augmentation')
    return data_augmentation
