"""Example of using IndexPreprocessor."""

import matplotlib.pyplot as plt

from preprocess import get_batch_data
from preprocess.indexed import IndexedPreprocessor
from preprocess.example.mnist import mnist_preprocessor, MnistDataset

if __name__ == '__main__':
    base = mnist_preprocessor(MnistDataset.TRAIN)
    preprocessor = IndexedPreprocessor(base)

    data = get_batch_data(preprocessor, batch_size=4, shuffle=True)
    for index, image, label in zip(*data):
        plt.imshow(image, cmap='gray')
        plt.title('Image %d, label %d' % (index, label))
        plt.show()
