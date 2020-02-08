import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .random_clip import RandomClip
from .random_peaking_filter import RandomPeakingFilter
from .random_time_shift import RandomTimeShift
from .random_time_stretch import RandomTimeStretch
from .random_noise import RandomNoise



class RandomAugmentation(object):
    """RandomAugmentation
    Apply transforms with random params
    Args:
        clip (dict)
        peaking_filter (dict)
        time_shift (dict)
        time_stretch (dict)
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                noise=None,
                clip=None,
                peaking_filter=None,
                time_shift=None,
                time_stretch=None,
                p=1.0):
        self.noise = noise
        self.clip = clip
        self.peaking_filter = peaking_filter
        self.time_shift = time_shift
        self.time_stretch = time_stretch
        self.p = p

    def __call__(self, data):
        x = data

        if self.noise is not None and np.random.rand() < self.p:
            x = RandomNoise(self.noise)(x)
        if self.clip is not None and np.random.rand() < self.p:
            x = RandomClip(self.clip)(x)
        if self.peaking_filter is not None and np.random.rand() < self.p:
            f = self.peaking_filter['f']
            q = self.peaking_filter['q']
            gain = self.peaking_filter['gain']
            x = RandomPeakingFilter(f, q, gain)(x)
        if self.time_shift is not None and np.random.rand() < self.p:
            x = RandomTimeShift(self.time_shift)(x)
        if self.time_stretch is not None and np.random.rand() < self.p:
            x = RandomTimeStretch(self.time_stretch)(x)

        return x
