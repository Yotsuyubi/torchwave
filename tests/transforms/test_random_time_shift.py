from torchwave.transforms import RandomTimeShift
import numpy as np
import pytest


def test_RandomTimeShift():
    noise = np.random.randn(512)
    assert RandomTimeShift()(noise).shape == (512,)
    assert RandomTimeShift(shift=30)(noise).shape == (512,)
    assert RandomTimeShift(shift=[40, 60])(noise).shape == (512,)
    s = RandomTimeShift(shift=[40, 60])
    _ = s(noise)
    assert 40 < s.shift < 60
    s = RandomTimeShift(shift=30)
    _ = s(noise)
    assert 0 < s.shift < 30

    with pytest.raises(ValueError):
        RandomTimeShift(shift=[40, 60, 1])(noise)
