"""mri_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import csv

# TODO(mri_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(mri_dataset): BibTeX citation
_CITATION = """
"""


class MriDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mri_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    # TODO: instructions manual download
    MANUAL_DOWNLOAD_INSTRUCTIONS = 'Placeholder'

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(512, 512, 3)),
                'label': tfds.features.ClassLabel(names=['0', '1', '2']),
                'id': tf.int32,
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # e.g. ('image', 'label')
            homepage='',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        extracted_path = dl_manager.manual_dir
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'images_dir_path': os.path.join(extracted_path, 'train'),
                    'labels': os.path.join(extracted_path, 'train_label.csv'),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'images_dir_path': os.path.join(extracted_path, 'test'),
                    'labels': None,
                },
            ),
        ]

    def _generate_examples(self, images_dir_path, labels=None):
        """Yields examples."""
        if labels is not None:
            with tf.io.gfile.GFile(labels) as f:
                for row in csv.DictReader(f):
                    image_id = row['ID']
                    # And yield (key, feature_dict)
                    yield image_id, {
                        'image': os.path.join(images_dir_path, f'{image_id}.png'),
                        'label': row['Label'],
                        'id': image_id,
                    }
        else:
            filenames = tf.io.gfile.listdir(path=images_dir_path)
            for filename in filenames:
                parts = tf.strings.split(filename, '.')
                image_id = int(parts[0].numpy())

                # And yield (key, feature_dict)
                yield image_id, {
                    'image': os.path.join(images_dir_path, f'{image_id}.png'),
                    'label': -1,  # NO LABEL
                    'id': image_id,
                }
