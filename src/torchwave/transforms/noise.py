import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable



class Noise(object):
    """Noise
    Add noise to samples
    Args:
        power (float): power of noise. Default: '0.1'
    Returns:
        numpy.ndarray with shape same as input's.
    """
    def __init__(self,
                power: float=0.1):
        self.power: float = power

    def __call__(self, data):
        noise = self.power*np.random.randn(len(data))
        return data+noise
