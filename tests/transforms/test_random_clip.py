import sys
sys.path.append('src')

from torchwave.transforms import RandomClip
import numpy as np
import pytest


def test_RandomClip():
    noise = np.random.randn(512)
    assert RandomClip()(noise).shape == (512,)
    assert RandomClip(m=0.6)(noise).shape == (512,)
    assert RandomClip(m=[0.5, 0.8])(noise).shape == (512,)

    with pytest.raises(ValueError):
        RandomClip(m=[0.5, 1.5, 1])(noise)
