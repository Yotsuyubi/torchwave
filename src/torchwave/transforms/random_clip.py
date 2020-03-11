import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .clip import Clip



class RandomClip(object):
    """RandomClip
    Apply Clip with randamized params
    """
    def __init__(self,
                m: Union[float, List[float]]=1.0):
        self.m: Union[float, List[float]] = m

    def _randamize_params(self):
        if type(self.m) is not list or len(self.m) < 2: # if rate given is float
            self.m = [0, self.m]

        if len(self.m) > 2:
            raise ValueError('given params too long.')

        self.m = self._rand_float(self.m[0], self.m[1])

    def __call__(self, data):
        self._randamize_params()
        return Clip(self.m)(data)

    def _rand_float(self, low, high):
        return (high - low) * np.random.rand() + low
