import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .noise import Noise



class RandomNoise(object):
    """RandomNoise
    Apply Noise with randamized params
    """
    def __init__(self,
                power: Union[float, List[float]]=.0):
        self.power: Union[float, List[float]] = power
        self._randamize_params()

    def _randamize_params(self):
        if type(self.power) is not list or len(self.power) < 2: # if rate given is float
            self.power = [0, self.power]

        if len(self.power) > 2:
            raise ValueError('given params too long.')

        self.power = self._rand_float(self.power[0], self.power[1])

    def __call__(self, data):
        return Noise(self.power)(data)

    def _rand_float(self, low, high):
        return (high - low) * np.random.rand() + low
