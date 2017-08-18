"""Provides a tensorflow preprocessor base class."""
import tensorflow as tf


class Preprocessor(object):
    """
    Class for preprocessing inputs for tensorflow queues.

    The inputs_fn passed to the constructor should return an identifier for
    each example in the dataset. For small datasets, this could be the actual
    dataset, e.g. for Mnist, inputs_fn could return [images, labels]. For
    datasets that are too big to fit in memory, an id could be used, e.g. for
    imagenet, inputs_fn could return a filenames tensor, assuming the class
    can be inferred from this.

    Primary use function is get_preprocessed_batch, which executes:
        1. get all examples (inputs) ->
        2. slice (single_inputs) ->
        3. apply per-example preprocessing (default nothing) ->
        4. batch (batch_inputs)

    Example usage:
    Using `tf.estimator.Estimator`:
    ```
    estimator = get_estimator()
    processor = get_preprocessor()
    batch_size = 128
    max_steps = 10000

    def train_input_fn():
        feature0, feature1, labels = preprocessor.get_preprocessed_batch(
            batch_size=batch_size, num_threads=8, shuffle=True)
        return (feature0, feature1), labels

    esimator.train(input_fn=fit_input_fn, max_steps=max_steps)

    def evaluate_input_fn():
        feature0, feature1, labels = preprocessor.get_preprocessed_batch(
            batch_size=batch_size, num_epochs=1, num_threds=8, shuffle=False)
        return (feature0, feature1), labels

    estimator.evaluate(input_fn=eval_input_fn)
    ```

    Without `tf.estimator.Estimator`s:
    ```
    processor = DerivedPreprocessor()
    graph = tf.Graph()
    with graph.as_default():
        feature0, feature1, labels = preprocessor.get_preprocessed_batch(
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
    """

    def __init__(self, inputs_fn):
        """
        Build tensors for start of preprocessing pipeline.

        Returns:
            list of tensors, each with same first dimension representing all
            possible examples.

        Example: consider an image classification problem with a numpy file
            `labels.npy` with label `i` corresponding to image
            `image_folder/i%i.jpg` % i. The input tensors would be a tensors
            or image paths and labels:
        ```
        from tensorflow.python.framework import ops

        labels_np = np.load('labels.npy')
        images_str = ['i%i.jpg' for i in range(len(labels_np))]
        labels_tf = ops.convert_to_tensor(
            labels_np, dtype=tf.uint8, name='labels')
        image_paths_tf = ops.convert_to_tensor(
            images_str, dtype=tf.string, name='image_paths')
        return image_paths_tf, labels_tf
        ```

        """
        self._inputs_fn = inputs_fn

    def inputs(self):
        """
        Build input tensors.

        Default implementation redirects to inputs_fn passed to constructor.
        """
        return self._inputs_fn()

    def single_inputs(self, inputs, num_epochs=None, shuffle=True):
        """
        Get a single record prior to preprocessing.

        Args:
            inputs: list/tuple of tensors or a single tensor representing all
                examples
            shuffle: whether or not to shuffle the input tensors (see
            `tf.train.slice_input_producer` `shuffle` arg).

        Returns:
            list of tensors representing a single example, or a single tensor
                if inputs is a single tensor.

        """
        def f(x):
            return tf.train.slice_input_producer(
                x, num_epochs=num_epochs, shuffle=shuffle)
        if isinstance(inputs, tf.Tensor):
            return f([inputs])[0]
        elif isinstance(inputs, dict):
            keys = [k for k in inputs]
            values = [inputs[k] for k in keys]
            values = f(values)
            return {k: v for (k, v) in zip(keys, values)}
        elif isinstance(inputs, (list, tuple)):
            return f(inputs)
        else:
            raise TypeError('inputs must be a tf.tensor, dict, list or tuple')

    def preprocess_single_inputs(self, single_inputs):
        """
        Preprocessing funcion applied to single examples.

        Can be overriden or changed via `map`. Defaults to identity.

        Args:
            single_inputs: list/tuple of inputs representing a single example.
        Returns:
            list/tuple of processed tensors representing a single example.

        Example usage:
            Given single_inputs (image_names, labels), where:
                image_name: tf.string tensor of image jpg names corresponding
                    to jpg files in `image_folder/image_name`
                label: tf.uint8 tensor of labels
            ```
            image_name, label = single_inputs
            image_folder = tf.constant(
                'image_folder/', dtype=tf.string, name='image_folder')
            image_data = tf.read(image_folder + image_name, name='image_data')
            image = tf.image.decode_jpeg(image_data, channels=3)
            image = some_tf_preprocessing_func(image)
            return image, label
            ```

        """
        return single_inputs

    def batch_inputs(
            self, single_inputs, batch_size, num_threads=1,
            allow_smaller_final_batch=False):
        """See `tf.train.batch`."""
        if isinstance(single_inputs, tf.Tensor):
            single_inputs = [single_inputs]
        return tf.train.batch(
            single_inputs, batch_size=batch_size, num_threads=num_threads,
            allow_smaller_final_batch=allow_smaller_final_batch)

    def get_preprocessed_batch(
            self, batch_size, num_epochs=None, shuffle=False, num_threads=8,
            allow_smaller_final_batch=False):
        """
        Combine standard preprocessing pipeline steps.

        Wraps:
            inputs,
            single_inputs,
            preprocess_single_inputs,
            batch_inputs

        """
        inputs = self.inputs()
        single_inputs = self.single_inputs(
            inputs, num_epochs=num_epochs, shuffle=shuffle)
        single_inputs = self.preprocess_single_inputs(single_inputs)
        batch = self.batch_inputs(
            single_inputs, batch_size=batch_size, num_threads=num_threads,
            allow_smaller_final_batch=allow_smaller_final_batch)
        return batch

    def map(self, map_fn):
        """
        Create a MappedPreprocessor using self as base.

        Returns self is map_fn is None.
        """
        if map_fn is None:
            return self
        else:
            return MappedPreprocessor(self, map_fn)

    def __str__(self):
        return 'Preprocessor'

    def __repr__(self):
        return self.__str__()


def get_batch_data(preprocessor, batch_size=4, shuffle=False, num_threads=4,
                   use_cpu=True):
    """
    Create the graph and run session, returning batch data.

    Session is closed before returning. Mostly for debugging/visualising data.

    Inputs:
        preprocessor: Preprocessor instance
        batch_size: size of batch returned
        shuffle: whether or not to have the preprocessor shuffle the data
        num_threads: number of threads working in parallel
        use_cpu: if True, constructs the graph entirely on /cpu:0.

    Returns:
        data associated with a preprocessed batch from the preprocessor.

    """
    def _build_graph():
        print('Building graph...')
        batch = preprocessor.get_preprocessed_batch(
            batch_size=batch_size, num_threads=num_threads, shuffle=shuffle)
        init = tf.global_variables_initializer()
        return batch, init

    graph = tf.Graph()
    with graph.as_default():
        if use_cpu:
            with tf.device('/cpu:0'):
                batch, init = _build_graph()
        else:
            batch, init = _build_graph()

    print('Running session...')
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            batch_data = sess.run(batch)
        except Exception:
            coord.request_stop()
            coord.join(threads)
            raise
        coord.request_stop()
        coord.join(threads)
    return batch_data


class MappedPreprocessor(Preprocessor):
    """A class representing the per-example mapping of a preprocessor."""

    def __init__(self, base_preprocessor, map_fn):
        if (map_fn is None):
            raise ValueError('map_fn cannot be None')
        self._base = base_preprocessor
        self._map_fn = map_fn

    def inputs(self):
        """Redirect to base preprocessor."""
        return self._base.inputs()

    def preprocess_single_inputs(self, *args, **kwargs):
        """
        Preprocessing function operating on individual example tensors.

        Applies the map_fn supplied in the constructor to the output of the
        base preprocessor's preprocess_single_inputs function.
        """
        return self._map_fn(
            self._base.preprocess_single_inputs(*args, **kwargs))

    def __str__(self):
        return 'Mapped:%s' % self._base

    def __repr__(self):
        return self.__str__()
