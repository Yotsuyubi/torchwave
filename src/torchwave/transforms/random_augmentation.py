import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .random_clip import RandomClip
from .random_crop import RandomCrop
from .random_peaking_filter import RandomPeakingFilter
from .random_time_shift import RandomTimeShift
from .random_time_stretch import RandomTimeStretch
from .repeat import Repeat



class RandomAugmentation(object):
    """RandomAugmentation
    Apply transforms with random params
    Args:
        crop (dict)
        clip (dict)
        peaking_filter (dict)
        time_shift (dict)
        time_stretch (dict)
    Returns:
        numpy.ndarray with shape (length).
    """
    def __init__(self,
                crop=None,
                clip=None,
                peaking_filter=None,
                time_shift=None,
                time_stretch=None):
        self.crop = crop
        self.clip = clip
        self.peaking_filter = peaking_filter
        self.time_shift = time_shift
        self.time_stretch = time_stretch

    def __call__(self, data):
        x = data

        if self.crop is not None:
            x = RandomCrop(self.crop)(x)
        if self.clip is not None:
            x = RandomClip(self.clip)(x)
        if self.peaking_filter is not None:
            f = self.peaking_filter['f']
            q = self.peaking_filter['q']
            gain = self.peaking_filter['gain']
            x = RandomPeakingFilter(f, q, gain)(x)
        if self.time_shift is not None:
            x = RandomTimeShift(self.time_shift)(x)
        if self.time_stretch is not None:
            x = RandomTimeStretch(self.time_stretch)(x)

        return x
