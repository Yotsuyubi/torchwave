from torchwave.transforms import RandomNoise
import numpy as np
import pytest


def test_RandomNoise():
    noise = np.random.randn(512)
    assert RandomNoise()(noise).shape == (512,)
    assert RandomNoise(power=0.6)(noise).shape == (512,)
    assert RandomNoise(power=[0.5, 0.8])(noise).shape == (512,)

    with pytest.raises(ValueError):
        RandomNoise(power=[0.5, 1.5, 1])(noise)
