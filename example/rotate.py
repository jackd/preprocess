"""
Example of preprocessor that randomly augments MNIST images by rotation.

Requires cv2.
"""
import tensorflow as tf
import cv2
from preprocess.example.mnist import MnistPreprocessor


def _rotate_image(image, rotation_deg):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotation_deg, 1)
    return cv2.warpAffine(image, M, (cols, rows))


class RotatingPreprocessor(MnistPreprocessor):
    """MnistPreprocessor that rotates images by a random amount."""

    def __init__(
            self, dataset, include_indices=False, max_rotation_deg=15.):
        """
        Initialize with images and labels from a dataset and noise prop.

        Inputs:
            dataset: mnist dataset, with images and labels property
            include_indices: whether or not to include example indices. See
                `inputs` / `preprocess_single_inputs` for details.
            max_rotation_deg: maximum value or rotation for data augmentation.

        Raises:
            ValueError is noise_prop not in [0, 1].
        """
        self._max_rotation_degrees = float(max_rotation_deg)
        super(RotatingPreprocessor, self).__init__(
            dataset, include_indices=include_indices)

    def preprocess_single_image(self, image):
        """
        Randomly sets pixel values to zero based on `max_rotation_deg`.

        `max_rotation_deg` set in constructor.
        """
        m = self._max_rotation_degrees
        if m > 0:
            d = tf.random_uniform(
                shape=(), minval=-m, maxval=m, dtype=tf.float32,
                name='rotation')
            shape = image.shape
            image = tf.py_func(
                _rotate_image, [image, d], tf.float32, stateful=False,
                name='rotated_image')
            image.set_shape(shape)
        return image


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
        RotatingPreprocessor(
            'train', include_indices=False, max_rotation_deg=15.),
        'training dropped (0.5)',
        shuffle=True)
    vis_preprocessor(
        RotatingPreprocessor(
            'validation', include_indices=False, max_rotation_deg=0.0),
        'validation dropped (0.0)',
        shuffle=False)
