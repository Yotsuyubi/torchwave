import sys
sys.path.append('src')

from torchwave.transforms import RandomPeakingFilter
import numpy as np
import pytest
import math

def test_init_RandomPeakingFilter():
    assert RandomPeakingFilter(1000, 1, 3).fs == 22050
    assert RandomPeakingFilter(1000, 1, 3).normalize == True
    with pytest.raises(TypeError):
        RandomPeakingFilter()


def test_process_RandomPeakingFilter():
    noise = np.random.randn(22050)
    filtered = RandomPeakingFilter(1000, 1, 3)(noise)
    assert filtered.shape == (22050,)
    noise = np.random.randn(22050)
    filtered = RandomPeakingFilter([100, 1000], 3, [-5, 3])(noise)
    assert filtered.shape == (22050,)

    with pytest.raises(ValueError):
        RandomPeakingFilter([100, 1000, 0], [1, 3], [-5, 3])(noise)
        RandomPeakingFilter([100, 1000], [1, 3, 0], [-5, 3])(noise)
        RandomPeakingFilter([100, 1000], [1, 3], [-5, 3, 0])(noise)
