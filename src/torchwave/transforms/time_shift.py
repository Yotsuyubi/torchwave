import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable



class TimeShift(object):
    """TimeShift
    TimeShift samples
    Args:
        length (int): length for crop sample.
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                shift: int=0):
        self.shift: int = shift

    def __call__(self, data):
        return np.roll(data, self.shift)
