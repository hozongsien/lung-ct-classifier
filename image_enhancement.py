import os
import cv2 as cv
import tensorflow as tf
from shutil import copyfile


def preprocess_images(data_folder, src_folder, dst_folder, ds_type, clip_limit=4, tile_grid_size=(8, 8)):
    src_filepath = os.path.join(data_folder, src_folder, ds_type)
    dst_filepath = os.path.join(data_folder, dst_folder, ds_type)

    filenames = tf.io.gfile.listdir(path=src_filepath)
    for filename in filenames:
        img = cv.imread(os.path.join(src_filepath, filename))
        img = hist_norm(img)
        img = clahe(img, clip_limit, tile_grid_size)
        cv.imwrite(os.path.join(dst_filepath, filename), img)


def hist_norm(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    norm_gray_img = cv.equalizeHist(gray_img)
    norm_img = cv.cvtColor(norm_gray_img, cv.COLOR_GRAY2RGB)
    return norm_img


def clahe(img, clipLimit=4, tileGridSize=(40, 40)):
    clahe = cv.createCLAHE(clipLimit, tileGridSize)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cl_img = clahe.apply(gray_img)
    cl_img = cv.cvtColor(cl_img, cv.COLOR_GRAY2RGB)
    return cl_img


def main():
    data_folder = 'data'
    src_folder = 'raw'
    dst_folder = 'processed'
    train_folder = 'train'
    test_folder = 'test'
    train_label = 'train_label.csv'

    # hyperparams
    clip_limit = 4
    tile_grid_size = (8, 8)

    # pre-pre-process dataset
    print(f'Preprocessing {train_folder}')
    preprocess_images(data_folder, src_folder, dst_folder,
                      train_folder, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    print(f'Preprocessing {test_folder}')
    preprocess_images(data_folder, src_folder, dst_folder,
                      test_folder, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    print(f'Copying {train_label}')
    _ = copyfile(
        os.path.join(data_folder, src_folder, train_label),
        os.path.join(data_folder, dst_folder, train_label)
    )


if __name__ == "__main__":
    main()
