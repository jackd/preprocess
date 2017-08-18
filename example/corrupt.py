"""Example of preprocessor that augments MNIST images by blacking out px."""
import tensorflow as tf
from preprocess.example.mnist import mnist_preprocessor


def get_corrupt_fn(drop_prob):
    if drop_prob == 0:
        return None
    else:
        def corrupt(single_inputs):
            image, label = single_inputs
            drop = tf.to_float(tf.greater(
                tf.random_uniform(image.shape, dtype=tf.float32), drop_prob))
            return image * drop, label
    return corrupt


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

    train_preprocessor = mnist_preprocessor('train').map(get_corrupt_fn(0.5))

    vis_preprocessor(
        train_preprocessor,
        'training dropped (0.5)',
        shuffle=True)
    vis_preprocessor(
        mnist_preprocessor('validation').map(get_corrupt_fn(0.0)),
        'validation dropped (0.0)',
        shuffle=False)
