"""
Example implementation and usage of Preprocessor class for MNIST data.

See also `corrupting.py` and `rotating.py` for examples of how this class can
be extended to provide augmented data.
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.examples.tutorials.mnist import input_data

from preprocess import Preprocessor

_mnist = input_data.read_data_sets(
    os.path.join(os.path.realpath(os.path.dirname(__file__)), 'MNIST_data'),
    one_hot=False)


class MnistDataset(object):
    """Standard names for MNIST datasets."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


def get_images(dataset):
    """
    Get images from an mnist dataset.

    `dataset` must be one of ['train', 'validation', 'test'] or
    one or [MnistDataset.TRAIN, MnistDataset.VALIDATION, MnistDataset.TEST]
    """
    return _get_dataset(dataset).images


def get_labels(dataset):
    """
    Get labels from an mnist dataset.

    `dataset` must be one of ['train', 'validation', 'test'] or
    one or [MnistDataset.TRAIN, MnistDataset.VALIDATION, MnistDataset.TEST]
    """
    return _get_dataset(dataset).labels


def _get_dataset(dataset):
    if dataset == MnistDataset.TRAIN:
        d = _mnist.train
    elif dataset == MnistDataset.VALIDATION:
        d = _mnist.validation
    elif dataset == MnistDataset.TEST:
        d = _mnist.test
    else:
        raise ValueError(
            'dataset must be one of '
            '[MnistDataset.TRAIN, MnistDataset.VALIDATION, '
            'MnistDataset.TEST], got %s' % dataset)
    return d


class MnistPreprocessor(Preprocessor):
    """Preprocessor implementation for Mnist data."""

    def __init__(self, dataset):
        """
        Initialize with Mnist dataset.

        Inputs:
            dataset: mnist dataset, with images and labels property

        Raises:
            ValueError: if len(dataset.images) != len(dataset.labels)

        """
        d = _get_dataset(dataset)
        self._images = np.reshape(d.images, (-1, 28, 28))
        self._labels = d.labels
        if len(self._images) != len(self._labels):
            raise ValueError('images and labels must be the same length.')

    def inputs(self):
        """
        Create (indices), images, labels inputs.

        Returns `indices, images, labels` if `include_indices == True` in
        the constructor, else just returns `images, labels`.
        """
        images = ops.convert_to_tensor(
            self._images, dtype=tf.float32, name='images')
        labels = ops.convert_to_tensor(
            self._labels, dtype=tf.int32, name='labels')
        return images, labels

    def preprocess_single_image(self, image):
        """Preprocessing function for a single image. Defaults to identity."""
        return image

    def preprocess_single_inputs(self, single_inputs):
        """
        Preprocess single inputs by randomly corrupting images.

        if include_indices is true (from constructor), returns
            index, image, label
        else just returns
            image, label
        """
        single_inputs[0] = self.preprocess_single_image(single_inputs[0])
        return single_inputs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from preprocess import get_batch_data

    def vis_preprocessor(preprocessor, title, shuffle, batch_size=2):
        """Visualize n images from the preprocessor."""
        images, labels = get_batch_data(
            preprocessor, batch_size=batch_size, use_cpu=True, shuffle=shuffle)

        for image, label in zip(images, labels):
            plt.imshow(image, cmap='gray')
            plt.title('%s, %s' % (title, str(label)))
            plt.show()

    vis_preprocessor(
        MnistPreprocessor(MnistDataset.TRAIN),
        'training normal',
        shuffle=True)
