import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable



class Crop(object):
    """Crop
    Crop samples
    Args:
        length (int): length for crop sample.
        start (int, optional): index for start crop.
                              Default: `0`
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                length: int,
                *,
                start: int=0):
        self.length: int = length
        self.start: int = start

    def __call__(self, data):
        if self.start > len(data):
            raise IndexError('`start` out of range.')
        if self.start < 0 or self.length < 0:
            raise ValueError('`start` and `length` must be positive int.')
        return data[self.start:self.start+self.length]
