import tensorflow as tf
import tensorflow_addons as tfa
import os


class AdaptiveConcatPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaptiveConcatPooling, self).__init__()

    def call(self, x):
        output_size = (x.shape[1], x.shape[2])
        avg_pool = tfa.layers.AdaptiveAveragePooling2D(output_size)(x)
        max_pool = tfa.layers.AdaptiveMaxPooling2D(output_size)(x)
        return tf.concat([avg_pool, max_pool], axis=1)


def create_base_model(model_type, img_shape):
    if model_type == 'mobile':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )
    elif model_type == 'xception':
        base_model = tf.keras.applications.Xception(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet',
        )
    elif model_type == 'res':
        base_model = tf.keras.applications.ResNet152V2(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet',
        )
    else:
        raise RuntimeError(f'model_type={model_type} unsupported')

    base_model.trainable = False
    return base_model


def create_prediction_layer(head_type, num_classes, hyperparams):
    if head_type == 'standard':
        prediction = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(hyperparams['dropout']),
            tf.keras.layers.Dense(num_classes, name='logits'),
            # separate activation layer for mixed precision support
            tf.keras.layers.Activation(
                'softmax', dtype='float32', name='probs'),
        ], name='prediction')
    elif head_type == 'adaptive':
        prediction = tf.keras.Sequential([
            AdaptiveConcatPooling(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.75),
            tf.keras.layers.Dense(num_classes, name='logits'),
            tf.keras.layers.Activation(
                'softmax', dtype='float32', name='probs'),
        ], name='prediction')
    else:
        raise RuntimeError(f'head_type={head_type} unsupported')

    return prediction


def create_model(model_params, hyperparams):
    base_model = create_base_model(
        model_params['model_type'], model_params['image_shape'])
    prediction_layer = create_prediction_layer(
        model_params['head_type'], model_params['num_classes'], hyperparams)

    inputs = tf.keras.Input(shape=model_params['image_shape'])
    x = base_model(inputs, training=False)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def create_fine_tune_model(model, hyperparams):
    print("Number of layers in the base model: ", len(model.layers[1].layers))
    model.layers[1].trainable = True
    for layer in model.layers[1].layers[:hyperparams['fine_tune_at']]:
        layer.trainable = False

    return model


def load_ensmble_models():
    filenames = ['Xception09.1-1_fold',
                 'MobileNet03-1_fold', 'ResNet03-1_fold']
    models = []
    for i, name in enumerate(filenames):
        model = tf.keras.models.load_model(os.path.join('models', name, '.h5'))
        for layer in model.layers:
            layer.trainable = False
            layer._name = f'ensemble_{i}_{layer.name}'
        models.append(model)
    return models


def create_ensemble_prediction_layer():
    prediction = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, name='logits'),
        tf.keras.layers.Activation('softmax', dtype='float32', name='probs'),
    ], name='prediction')
    return prediction


def create_ensemble_model():
    models = load_ensmble_models()
    prediction_layer = create_ensemble_prediction_layer()

    ensemble_inputs = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    concat = tf.keras.layers.concatenate(ensemble_outputs)
    outputs = prediction_layer(concat)
    model = tf.keras.Model(inputs=ensemble_inputs, outputs=outputs)
    return model
