import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .crop import Crop



class RandomCrop(object):
    """RandomCrop
    Crop samples randamly
    Args:
        length (int): length for crop sample.
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                length: Optional[int]=None):
        self.length: int = length if length is not None else 0

    def __call__(self, data):
        start = np.random.randint(0, np.abs(len(data)-self.length))
        if self.length < 0:
            raise ValueError('`length` must be positive int.')
        if len(data)-self.length < 0:
            raise ValueError('`length` too large.')
        return Crop(length=self.length, start=start)(data)
