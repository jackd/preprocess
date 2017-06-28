"""Provides `SequencePreprocessor` for sampling sub-sequences."""
import tensorflow as tf
from tensorflow.python.framework import ops
from preprocessor import Preprocessor
#
#
# def _sequence_length(sequence):
#     if isinstance(sequence, tf.Tensor):
#         return sequence.shape.as_list()[0]
#     elif isinstance(sequence, (list, tuple)):
#         return _sequence_length(sequence[0])
#     elif isinstance(sequence, dict):
#         for k in sequence:
#             return _sequence_length(sequence[k])
#     else:
#         raise Exception(
#             'Cannot get length of sequence of type %s' % type(sequence))
#
#
# def _merge_sequences(s0, s1):
#     assert(type(s0) == type(s1))
#     if isinstance(s0, np.ndarray):
#
#     if isinstance(s0, (list, tuple)):


class SubsequencePreprocessor(Preprocessor):
    """
    Partial Preprocessor implementation for extracting subsequences.

    `get_sequences()` must be implemented.
    """

    def __init__(self, subsequence_length):
        self._subsequence_length = subsequence_length

    def get_sequence_data(self):
        """
        Get full sequences.

        Must be implemented by derived class

        Returns data, lengths.

        data: dictionary mapping keys to concatenated sequence tensors. Each
            value must be the same length in the first dimension.
        lengths: iterable of ints denoting length of each full sequence.

        sum(lengths) == data[any_key].shape[0]

        e.g.
        data = {
            'a': ops.convert_to_tensor(
                [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 5], dtype=tf.int32)
            'b': ops.convert_to_tensor(
                [10, 9, 8, 7, 10, 9, 8, 10, 9, 8, 7, 6, 5], dtype=tf.int32)
        },
        lengths = [4, 3, 6]
        """
        raise NotImplementedError()

    def _build_starts(self, lengths):
        """Build starts tensor."""
        sublength = self._subsequence_length
        start = 0
        starts = []
        for length in lengths:
            n = max(length - sublength+1, 0)
            starts.extend(range(start, start + n))
            start += length
        return ops.convert_to_tensor(starts, dtype=tf.int32)

    def inputs(self):
        """Get raw inputs. Just starts. Also calls self.get_sequences()."""
        self._data, lengths = self.get_sequence_data()
        self._starts = self._build_starts(lengths)
        return self._starts

    def preprocess_single_inputs(self, single_inputs):
        """
        Convert a start value to a subsequence.

        Returns a dict mapping keys from get_sequences data arg
        """
        start = single_inputs
        data = {k: v[start: start + self._subsequence_length]
                for k, v in self._data.items()}
        for k, v in data.items():
            v.set_shape(
                [self._subsequence_length] + self._data[k].shape.as_list()[1:])
        return data


if __name__ == '__main__':
    from preprocess import get_batch_data

    class TestSubsequencePreprocessor(SubsequencePreprocessor):
        def get_sequence_data(self):
            lengths = [5, 4, 7, 5]
            vals = []
            for i, length in enumerate(lengths):
                vals.extend(range(i*10, i*10 + length))
            data = {'vals': ops.convert_to_tensor(vals, dtype=tf.int32)}
            return data, lengths

    preprocessor = TestSubsequencePreprocessor(subsequence_length=5)
    vals = get_batch_data(preprocessor, batch_size=4, shuffle=True)['vals']
    print(vals)
