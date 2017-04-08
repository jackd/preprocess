"""Provides IndexedPreprocessor."""
import tensorflow as tf
from preprocess import Preprocessor


def _batch_len(inputs):
    if isinstance(inputs, (list, tuple)):
        return inputs[0].shape[0]
    elif isinstance(inputs, dict):
        for k, v in inputs.items():
            break
        return v.shape[0]
    elif isinstance(inputs, tf.Tensor):
        return inputs.shape[0]
    else:
        raise TypeError('inputs must be list, tuple, dict or tf.Tensor.')


class IndexedPreprocessor(Preprocessor):
    """Wrapper for another preprocessor that adds an index."""

    def __init__(self, base_preprocessor, index_key='index'):
        """
        Initialize with a preprocessor to wrap.

        `index_key` will be used if the output of any methods is a dict,
        otherwise is ignored.
        """
        self._base_preprocessor = base_preprocessor
        self._index_key = index_key

    @property
    def base_preprocessor(self):
        """Get the wrapped preprocessor."""
        return self._base_preprocessor

    def _combine(self, index, original):
        if isinstance(original, list):
            return [index] + original
        elif isinstance(original, tuple):
            return (index,) + original
        elif isinstance(original, dict):
            if self._index_key in original:
                raise KeyError(
                    'index key supplied in constructor already in original.')
            original[self._index_key] = index
            return original
        else:
            raise TypeError('original must be list, tuple, dict or Tensor.')

    def _split(self, original):
        if isinstance(original, (list, tuple)):
            return original[0], original[1:]
        elif isinstance(original, dict):
            if self._index_key not in original:
                raise KeyError(
                    'index key supplied in constructor not in original.')
            index = original[self._index_key]
            del original[self._index_key]
            return index, original
        else:
            raise TypeError('original must be list, tuple or dict.')

    def inputs(self):
        """See Preprocessor.inputs."""
        original = self._base_preprocessor.inputs()
        n = _batch_len(original)
        index = tf.range(n, dtype=tf.int32, name='index_all')
        return self._combine(index, original)

    def preprocess_single_inputs(self, single_inputs):
        """See Preprocessor.preprocess_single_inputs."""
        index, original = self._split(single_inputs)
        original = self._base_preprocessor.preprocess_single_inputs(original)
        return self._combine(index, original)
