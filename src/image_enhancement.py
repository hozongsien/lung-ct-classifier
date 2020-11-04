import argparse
import os
import cv2 as cv
import tensorflow as tf
from shutil import copyfile


def preprocess_images(src_path, dst_path, ds_type, clip_limit=4, tile_grid_size=(8, 8)):
    """Applies image pre-processing on the specified data."""
    src_filepath = os.path.join(src_path, ds_type)
    dst_filepath = os.path.join(dst_path, ds_type)

    filenames = tf.io.gfile.listdir(path=src_filepath)
    for filename in filenames:
        img = cv.imread(os.path.join(src_filepath, filename))
        img = hist_norm(img)
        img = clahe(img, clip_limit, tile_grid_size)
        cv.imwrite(os.path.join(dst_filepath, filename), img)


def hist_norm(img):
    """Applies histogram normalisation on the given image."""
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    norm_gray_img = cv.equalizeHist(gray_img)
    norm_img = cv.cvtColor(norm_gray_img, cv.COLOR_GRAY2RGB)
    return norm_img


def clahe(img, clipLimit=4, tileGridSize=(40, 40)):
    """Applies contrast limited histogram normalisation on the given image."""
    clahe = cv.createCLAHE(clipLimit, tileGridSize)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cl_img = clahe.apply(gray_img)
    cl_img = cv.cvtColor(cl_img, cv.COLOR_GRAY2RGB)
    return cl_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str,
                        help='source path of the original data.')
    parser.add_argument('dst_path', type=str,
                        help='destination path of the data after enhancement.')
    args = parser.parse_args()

    train_folder = 'train'
    test_folder = 'test'
    train_label = 'train_label.csv'

    # hyperparams
    clip_limit = 4
    tile_grid_size = (8, 8)

    # enhance image dataset
    print(f'Preprocessing {train_folder}')
    preprocess_images(args.src_path, args.dst_path, train_folder,
                      clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    print(f'Preprocessing {test_folder}')
    preprocess_images(args.src_path, args.dst_path, test_folder,
                      clip_limit=clip_limit, tile_grid_size=tile_grid_size)

    print(f'Copying {train_label}')
    _ = copyfile(
        os.path.join(args.src_path, train_label),
        os.path.join(args.dst_path, train_label)
    )


if __name__ == "__main__":
    main()
