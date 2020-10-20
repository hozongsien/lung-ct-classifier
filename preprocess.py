import cv2 as cv
import os
import tensorflow as tf
from shutil import copyfile


def preprocess_images(data_folder, raw_folder, processed_folder, ds_type, clip_limit=4, tile_grid_size=(40, 40)):
    src_filepath = os.path.join(data_folder, raw_folder, ds_type)
    dst_filepath = os.path.join(data_folder, processed_folder, ds_type)

    filenames = tf.io.gfile.listdir(path=src_filepath)
    for filename in filenames:
        img = cv.imread(os.path.join(src_filepath, filename))
        img = clahe(img, clip_limit, tile_grid_size)
        cv.imwrite(os.path.join(dst_filepath, filename), img)


def clahe(img, clipLimit=4, tileGridSize=(40, 40)):
    clahe = cv.createCLAHE(clipLimit, tileGridSize)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cl_img = clahe.apply(gray_img)
    cl_img = cv.cvtColor(cl_img, cv.COLOR_GRAY2RGB)
    return cl_img


# pre-pre-process dataset
data_folder = 'data'
raw_folder = 'raw'
processed_folder = 'processed'
train_folder = 'train'
test_folder = 'test'
train_label = 'train_label.csv'

print(f'Preprocessing {train_folder}')
preprocess_images(data_folder, raw_folder, processed_folder,
                  train_folder, clip_limit=4, tile_grid_size=(40, 40))
print(f'Preprocessing {test_folder}')
preprocess_images(data_folder, raw_folder, processed_folder,
                  test_folder, clip_limit=4, tile_grid_size=(40, 40))
print(f'Copying {train_label}')
_ = copyfile(
    os.path.join(data_folder, raw_folder, train_label),
    os.path.join(data_folder, processed_folder, train_label)
)
