from typing import Callable
import numpy as np

class RandomApply(object):
    """RandomApply
    Apply transform randamly
    Args:
        0 < p (float) < 1: Probability.
        transform (Callable): transform to apply.
    Returns:
        numpy.ndarray[numpy.float]
    """

    def __init__(self, p, transform):
        self.p: float = p
        self.transform: Callable[[np.ndarray[np.float32]], np.ndarray[np.float32]] = transform

    def __call__(self, data):
        if 0 > self.p or self.p > 1:
            raise ValueError('`p` must be 0 < p < 1.')
        randamly_transformed = self.transform(data) if np.random.randint(0, 1) < self.p else data
        return randamly_transformed
