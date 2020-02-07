import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable



class Clip(object):
    """Clip
    Clip samples
    Args:
        m (float): max/min.
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                m: float=1.0):
        self.m: float = m

    def __call__(self, data):
        if self.m < 0:
            raise ValueError('`m` must be positive int.')
        return np.clip(data, -self.m, self.m)
