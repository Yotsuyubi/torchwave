import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .time_shift import TimeShift



class RandomTimeShift(object):
    """RandomTimeShift
    Apply TimeShift with randamized params
    """
    def __init__(self,
                shift: Union[None, List[int]]=None):
        self.shift: Union[None, List[int]] = shift

    def _randamize_params(self):
        if type(self.shift) is not list or len(self.shift) < 2: # if rate given is float
            self.shift = [0, self.shift]

        if len(self.shift) > 2:
            raise ValueError('given params too long.')

        self.shift = np.random.randint(self.shift[0], self.shift[1])

    def __call__(self, data):
        if self.shift is None:
            self.shift = len(data)
        self._randamize_params()
        return TimeShift(self.shift)(data)
