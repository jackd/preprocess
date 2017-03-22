"""Example of preprocessor that augments MNIST images by blacking out px."""
import tensorflow as tf
from preprocess.example.mnist import MnistPreprocessor


class CorruptingPreprocessor(MnistPreprocessor):
    """MnistPreprocessor that drops pixels at random."""

    def __init__(self, dataset, include_indices=False, drop_prop=0.5):
        """
        Initialize with images and labels from a dataset and drop prop.

        Inputs:
            dataset: mnist dataset, with images and labels property
            include_indices: whether or not to include example indices. See
                `inputs` / `preprocess_single_inputs` for details.
            drop_prop: probability of setting each pixel to one.

        Raises:
            ValueError is noise_prop not in [0, 1].
        """
        if drop_prop < 0 or drop_prop > 1:
            raise ValueError('noise_prop must be in range [0, 1]')
        self._drop_prop = drop_prop
        super(CorruptingPreprocessor, self).__init__(
            dataset, include_indices=include_indices)

    def preprocess_single_image(self, image):
        """
        Randomly sets pixel values to zero based on drop_prop.

        `drop_prop` set in constructor.
        """
        if self._drop_prop > 0:
            drop = tf.to_float(tf.greater(
                tf.random_uniform(shape=image.shape, dtype=tf.float32),
                self._drop_prop, name='dropped'))
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
        CorruptingPreprocessor('train', False, 0.5),
        'training dropped (0.5)',
        shuffle=True)
    vis_preprocessor(
        CorruptingPreprocessor('validation', False, 0.0),
        'validation dropped (0.0)',
        shuffle=False)
