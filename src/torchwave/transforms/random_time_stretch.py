import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .time_stretch import TimeStretch



class RandomTimeStretch(object):
    """RandomTimeStretch
    Apply Time Stretch with randamized params
    """
    def __init__(self,
                rate: Union[float, List[float]]=1):
        self.rate: Union[float, List[float]] = rate
        self._randamize_params()

    def _randamize_params(self):
        if type(self.rate) is not list or len(self.rate) < 2: # if rate given is float
            self.rate = [0, self.rate]

        if len(self.rate) > 2:
            raise ValueError('given params too long.')

        self.rate = self._rand_float(self.rate[0], self.rate[1])

    def __call__(self, data):
        return TimeStretch(self.rate)(data)

    def _rand_float(self, low, high):
        return (high - low) * np.random.rand() + low
