from torchwave.transforms import PeakingFilter
import numpy as np
import pytest
import math

def test_init_PeakingFilter():
    assert PeakingFilter(1000, 1, 3).fs == 22050
    assert PeakingFilter(1000, 1, 3).normalize == True
    with pytest.raises(TypeError):
        PeakingFilter()


def test_process_PeakingFilter():
    noise = np.random.randn(22050)
    filtered = PeakingFilter(1000, 1, 3)(noise)
    assert filtered.shape == (22050,)

    with pytest.raises(ValueError):
        PeakingFilter(-0.1, 1, 3)(noise)
        PeakingFilter(1000, -0.1, 3)(noise)
        PeakingFilter(1000, 25, 3)(noise)
        PeakingFilter(1000, 1, -25)(noise)
        PeakingFilter(1000, 1, 25)(noise)
