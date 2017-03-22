"""Provides a tensorflow preprocessor base class."""
import tensorflow as tf


class Preprocessor(object):
    """
    Base class for preprocessing inputs for tensorflow queues.

    Derived classes must implement:
        inputs(self)

    Derived classes may implement:
        preprocess_single_inputs(self, single_inputs)
        preprocess_batch_inputs(self, batch_inputs)

    (default implementation is identity)

    Primary use function is get_preprocessed_batch, which executes:
        get all examples (inputs) ->
        slice (single_inputs) ->
        preprocess single inputs (preprocessed_single_inputs) ->
        batch (batch_inputs)

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
    """

    def inputs(self):
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
        raise NotImplementedError()

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
        else:
            return f(inputs)

    def preprocess_single_inputs(self, single_inputs):
        """
        Preprocessing funcion applied to single examples.

        Args:
            single_inputs: list/tuple of inputs representing a single example.
        Returns:
            list/tuple of processed tensors representing a single example.

        Defaults to identity.

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
            self, single_inputs, batch_size, num_threads=1):
        """See `tf.train.batch`."""
        if isinstance(single_inputs, tf.Tensor):
            single_inputs = [single_inputs]
        return tf.train.batch(
            single_inputs, batch_size=batch_size, num_threads=num_threads)

    def get_preprocessed_batch(
            self, batch_size, num_epochs=None, shuffle=False, num_threads=8):
        """
        Convenience function.

        Wraps:
            inputs,
            single_inputs,
            preprocessed_single_inputs,
            batch_inputs
        """
        inputs = self.inputs()
        single_inputs = self.single_inputs(
            inputs, num_epochs=num_epochs, shuffle=shuffle)
        single_inputs = self.preprocess_single_inputs(single_inputs)
        batch = self.batch_inputs(
            single_inputs, batch_size=batch_size, num_threads=num_threads)
        return batch


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
        except:
            coord.request_stop()
            coord.join(threads)
            raise
        coord.request_stop()
        coord.join(threads)
    return batch_data
