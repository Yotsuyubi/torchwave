import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .peaking_filter import PeakingFilter



class RandomPeakingFilter(object):
    """RandomPeakingFilter
    Apply peaking filter with randamized params
    """
    def __init__(self,
                f: Union[float, List[float]],
                q: Union[float, List[float]],
                gain: Union[float, List[float]],
                *,
                fs: int=22050,
                normalize: bool=True):
        self.f: Union[float, List[float]] = f
        self.q: Union[float, List[float]] = q
        self.gain: Union[float, List[float]] = gain
        self.fs: int = fs
        self.normalize: bool = True

    def _randamize_params(self):
        if type(self.f) is not list or len(self.f) < 2: # if f given is float
            self.f = [0, self.f]
        if type(self.q) is not list or len(self.q) < 2: # if q given is float
            self.q = [0.01, self.q]
        if type(self.gain) is not list or len(self.gain) < 2: # if gain given is float
            self.gain = [-self.gain, self.gain]

        if len(self.f) > 2 or len(self.q) > 2 or len(self.gain) > 2:
            raise ValueError('given params too long.')

        self.f = self._rand_float(self.f[0], self.f[1])
        self.q = self._rand_float(self.q[0], self.q[1])
        self.gain = self._rand_float(self.gain[0], self.gain[1])

    def __call__(self, data):
        self._randamize_params()
        return PeakingFilter(self.f,
                             self.q,
                             self.gain,
                             fs=self.fs,
                             normalize=self.normalize)(data)

    def _rand_float(self, low, high):
        return (high - low) * np.random.rand() + low
