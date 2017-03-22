# preprocess

Provides the `Preprocessor` class for preprocessing data for `tensorflow`.

## Requirements
This module is designed for use with [Tensorflow](https://www.tensorflow.org/)

The following modules are required in order to run the specified examples:
* [OpenCV](http://opencv.org/) (`example/rotate.py`)
* [Pillow](https://github.com/python-pillow/Pillow) (`example/load.py`)
* [Numpy](http://www.numpy.org/) (`example/load.py`)

There is currently no install script. To use this module from outside the folder or the examples, add the parent directory to your `PYTHONPATH`. For example, if you downloaded and extracted/cloned this repository to `/path/to/preprocess`, run

```
export PYTHONPATH=$PYTHONPATH:/path/to
```

You may wish to append this line to your `~/.bashrc` file.

## Usage
Machine learning methods require large amounts of data to learn, and this data often needs preprocessing or manipulation each time it is passed into a model. The `Preprocessor` class provides a framework to do this data preprocessing and augmentation in a simple and consistent manner.

Example usage:

Using `tf.contrib.learn.Estimator`:
```
estimator = get_estimator()
processor = DerivedPreprocessor()
batch_size = 128
max_steps = 10000

def fit_input_fn():
    feature0, feature1, labels = preprocessor.get_preprocessed_batch(
        batch_size=batch_size, num_threads=8, shuffle=True)
    return (feature0, feature1), labels

esimator.fit(input_fn=fit_input_fn, max_steps=max_steps)

def evaluate_input_fn():
    feature0, feature1, labels = preprocessor.get_preprocessed_batch(
        batch_size=batch_size, num_epochs=1, num_threds=8, shuffle=False)
    return (feature0, feature1), labels

estimator.evaluate(input_fn=input_fn)
```

Without `tf.contrib.learn.Estimator`s:
```
processor = DerivedPreprocessor()
graph = tf.Graph()
with graph.as_default():
    feature0, feature1, labels = get_preprocessed_batch(
        batch_size=128, num_threads=8, shuffle=True)
    inference = get_inference(feature0, feature1)
    loss = get_loss(inference, labels)
    opt = get_opt(loss)
    init = tf.global_variables_initializer()

with tf.Session(graph=graph as sess):
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while True:
            loss_value, _ = sess.run([loss, opt])
    except:
        coord.request_stop()
        coord.join(threads)
        raise
    coord.request_stop()
    coord.join(threads)
```

## Implementating your own Preprocessor
Each derived class is only required to implement the `inputs(self)` function, which should return a `Tensor` or `list`, `tuple` or string-keyed`dict` or tensors.

See `example.mnist` for a minimal implementation based on the MNIST dataset.

See `example.load` for an example where images are loaded from file as required (necessary if there are too many inputs to fit in memory for example).

Optionally, a `preprocess_single_inputs(self, single_inputs)` can be implemented
such that each time an example is used by the model it will undergo some preprocessing. See `example.corrupt` and `example.rotate` for examples of random image corruption and rotation respectively.

## Testing/visualising your preprocessor
The function `get_batch_data` is also provided to make visualising output easier. An example of this can be seen in each of the example scripts.

```
cd example
python mnist.py
```

## Using a Preprocessor with an Estimator.
A complete example including training, evaluation and predictions using the MNIST `CorruptingPreprocessor` (`example.corrupt`) with a custom `tf.contrib.learn.Estimator` can be seen [here](https://github.com/jackd/mnist_estimator).
