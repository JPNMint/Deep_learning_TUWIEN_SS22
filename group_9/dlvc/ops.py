import random
from typing import List, Callable

import numpy as np

# All operations are functions that take and return numpy arrays
# See https://docs.python.org/3/library/typing.html#typing.Callable for what this line means
Op = Callable[[np.ndarray], np.ndarray]

def chain(ops: List[Op]) -> Op:
    '''
    Chain a list of operations together.
    '''

    def op(sample: np.ndarray) -> np.ndarray:
        for op_ in ops:
            sample = op_(sample)
        return sample

    return op

def type_cast(dtype: np.dtype) -> Op:
    '''
    Cast numpy arrays to the given type.
    '''

    return lambda sample: sample.astype(dtype)

def vectorize() -> Op:
    '''
    Vectorize numpy arrays via "numpy.ravel()".
    '''

    return lambda sample: sample.ravel()

def add(val: float) -> Op:
    '''
    Add a scalar value to all array elements.
    '''

    return lambda sample: sample + val

def mul(val: float) -> Op:
    '''
    Multiply all array elements by the given scalar.
    '''

    return lambda sample: sample * val

def hwc2chw() -> Op:
    '''
    Flip a 3D array with shape HWC to shape CHW.
    '''

    return lambda sample: np.transpose(sample, (2, 0, 1))

def hflip() -> Op:
    '''
    Flip arrays with shape HWC horizontally with a probability of 0.5.
    '''

    return lambda sample: np.flip(sample, axis=0) if np.random.random() < 0.5 else sample


def rcrop(sz: int, pad: int, pad_mode: str) -> Op:
    '''
    Extract a square random crop of size sz from arrays with shape HWC.
    If pad is > 0, the array is first padded by pad pixels along the top, left, bottom, and right.
    How padding is done is governed by pad_mode, which should work exactly as the 'mode' argument of numpy.pad.
    Raises ValueError if sz exceeds the array width/height after padding.
    '''

    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html will be helpful
    def _rcrop(sample: np.ndarray) -> np.ndarray:
        if pad > 0:
            sample = np.pad(sample, ((pad, pad), (pad, pad), (0, 0)), pad_mode)

        if sample.shape[0] < sz:
            raise ValueError(f"Invalid value for argument 'sz'. Actual: {sz}. Expected: <= {sample.shape[0]}.")

        if sample.shape[1] < sz:
            raise ValueError(f"Invalid value for argument 'sz'. Actual: {sz}. Expected: <= {sample.shape[1]}.")

        y = np.random.randint((sample.shape[0] - sz) + 1)
        x = np.random.randint((sample.shape[1] - sz) + 1)

        return sample[y:y+sz, x:x+sz]

    return _rcrop


def normalizePerChannel(mean: np.ndarray, std: np.ndarray) -> Op:
    """
    Normalizes an image per channel, i.e., (I - mean)/std where I is an image in (C, H, W) format.
    """
    if not isinstance(mean, np.ndarray):
        raise TypeError(f"Argument 'mean' has invalid type. Actual: {type(mean)}. Exepcted: {np.ndarray}.")

    if not isinstance(std, np.ndarray):
        raise TypeError(f"Argument 'std' has invalid type. Actual: {type(std)}. Exepcted: {np.ndarray}.")

    if mean.dtype != np.float32:
        raise ValueError(f"Argument 'mean' has invalid dtype. Actual: {mean.dtype}. Expected: {np.float32}.")

    if std.dtype != np.float32:
        raise ValueError(f"Argument 'std' has invalid dtype. Actual: {std.dtype}. Expected: {np.float32}.")

    return lambda sample: (sample - mean)/std
