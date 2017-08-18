"""
Example of preprocessor that randomly augments MNIST images by rotation.

Uses tf.py_func to augment via standard python function.

Requires cv2.
"""
import tensorflow as tf
import cv2
from preprocess.example.mnist import mnist_preprocessor


def _rotate_image(image, rotation_deg):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rotation_deg, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def get_rotation_fn(max_rotation_deg=15.):
    if max_rotation_deg == 0:
        return None

    def rotate(inputs):
        image, labels = inputs
        d = tf.random_uniform(
            shape=(), minval=-max_rotation_deg, maxval=max_rotation_deg,
            dtype=tf.float32, name='rotation')
        shape = image.shape
        image = tf.py_func(
            _rotate_image, [image, d], tf.float32, stateful=False,
            name='rotated_image')
        image.set_shape(shape)
        return image, labels

    return rotate


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
        mnist_preprocessor('train').map(get_rotation_fn(45.)),
        'training rotated(45.)',
        shuffle=True)
    vis_preprocessor(
        mnist_preprocessor('validation').map(get_rotation_fn(0)),
        'validation rotated (0.0)',
        shuffle=False)
