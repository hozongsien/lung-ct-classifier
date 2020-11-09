import argparse
import os
import cv2 as cv
import tensorflow as tf
from shutil import copyfile


def preprocess_images(src_path, dst_path, clip_limit=4, tile_grid_size=(8, 8)):
    """Applies image pre-processing on the specified data."""
    filenames = tf.io.gfile.listdir(path=src_path)
    for filename in filenames:
        img = cv.imread(os.path.join(src_path, filename))
        img = clahe(img, clip_limit, tile_grid_size)
        cv.imwrite(os.path.join(dst_path, filename), img)


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
    parser.add_argument('src_data_path', type=str,
                        help='Read location of raw data.')
    parser.add_argument('dst_data_path', type=str,
                        help='Write location of processed data.')

    args = parser.parse_args()

    src_train_data_path = 'train_image/train_image'
    src_test_data_path = 'test_image/test_image'
    dst_train_data_path = 'train'
    dst_test_data_path = 'test'
    train_label = 'train_label.csv'

    # hyperparams
    clip_limit = 8
    tile_grid_size = (32, 32)

    # enhance images
    print(f'Preprocessing train images...')
    preprocess_images(
        src_path=os.path.join(args.src_data_path, src_train_data_path),
        dst_path=os.path.join(args.dst_data_path, dst_train_data_path),
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size
    )

    print(f'Preprocessing test images...')
    preprocess_images(
        src_path=os.path.join(args.src_data_path, src_test_data_path),
        dst_path=os.path.join(args.dst_data_path, dst_test_data_path),
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size
    )

    print(f'Copying train labels...')
    _ = copyfile(
        src=os.path.join(args.src_data_path, train_label),
        dst=os.path.join(args.dst_data_path, train_label)
    )


if __name__ == "__main__":
    main()
