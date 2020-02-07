from torchwave.transforms import Clip
import numpy as np
import pytest
import math

def test_init_Clip():
    assert Clip().m == 1.0


def test_process_Clip():
    noise = np.random.randn(2048)
    clipped = Clip(m=0.5)(noise)
    assert clipped.shape == (2048,)
    assert np.max(clipped) == 0.5
    assert np.min(clipped) == -0.5

    with pytest.raises(ValueError):
        Clip(m=-0.5)(noise)
