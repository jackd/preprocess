"""
Example of using a preprocessor to load an image from file.

Useful when examples don't all fit in memory.

Requires PIL.Image if data is to be generated. Otherwise, requires some images
in `saved_images` directory.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from preprocess import Preprocessor

images_dir = os.path.join(os.path.dirname(__file__), 'saved_images')


def has_images():
    """Get a bool, True is there are images in `images_dir`."""
    return os.path.isdir(images_dir) and len(os.listdir(images_dir)) != 0


def image_path(index):
    """Get the path to `index`th image."""
    return os.path.join(images_dir, '%d.jpg' % index)


def generate_images(n=10):
    """Generate rubbish images for demonstration purposes only."""
    from PIL import Image
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    if len(os.listdir(images_dir)) == 0:
        for index in range(n):
            data = (np.random.random((128, 128, 3))*255).astype(np.uint8)
            path = image_path(index)
            Image.fromarray(data).save(path)


class LoadingPreprocessor(Preprocessor):
    """
    Preprocessor implementation that loads data from file.

    Each example's data is in it's own seperate file.
    """

    def inputs(self):
        """Get a tensor representing all image filenames."""
        filenames = list(os.listdir(images_dir))
        filenames = ops.convert_to_tensor(
            filenames, dtype=tf.string, name='filenames')
        return filenames

    def preprocess_single_inputs(self, single_inputs):
        """Convert filenames into images."""
        filename = single_inputs
        path = ('%s/' % images_dir) + filename
        image_data = tf.read_file(path, name='image_data')
        image = tf.image.decode_jpeg(image_data, channels=3, name='image')
        image.set_shape((128, 128, 3))
        return image


if __name__ == '__main__':
    from preprocess import get_batch_data
    import matplotlib.pyplot as plt
    if not has_images():
        generate_images()
    preprocessor = LoadingPreprocessor()
    images = get_batch_data(preprocessor, batch_size=4)
    print(images.shape)
    for image in images:
        plt.imshow(image)
        plt.show()
