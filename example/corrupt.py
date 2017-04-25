"""Example of preprocessor that augments MNIST images by blacking out px."""
import tensorflow as tf
from preprocess.example.mnist import MnistPreprocessor


class CorruptingPreprocessor(MnistPreprocessor):
    """MnistPreprocessor that drops pixels at random."""

    def __init__(self, dataset, drop_prob=0.5):
        """
        Initialize with images and labels from a dataset and drop prop.

        Inputs:
            dataset: mnist dataset, with images and labels property
            drop_prob: probability of setting each pixel to one.

        Raises:
            ValueError is noise_prop not in [0, 1].
        """
        if drop_prob < 0 or drop_prob > 1:
            raise ValueError('noise_prob must be in range [0, 1]')
        self._drop_prob = drop_prob
        super(CorruptingPreprocessor, self).__init__(dataset)

    def preprocess_single_image(self, image):
        """
        Randomly sets pixel values to zero based on `drop_prob`.

        `drop_prob` set in constructor.
        """
        if self._drop_prob > 0:
            drop = tf.to_float(tf.greater(
                tf.random_uniform(shape=image.shape, dtype=tf.float32),
                self._drop_prob, name='dropped'))
            image = image * drop
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
        CorruptingPreprocessor('train', 0.5),
        'training dropped (0.5)',
        shuffle=True)
    vis_preprocessor(
        CorruptingPreprocessor('validation', 0.0),
        'validation dropped (0.0)',
        shuffle=False)
