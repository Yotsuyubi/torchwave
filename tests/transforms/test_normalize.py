from torchwave.transforms.normalize import Normalize
import numpy as np
import pytest

def test_Normalize():
    noise = np.random.randn(512)
    assert Normalize()(noise).shape == (512,)
    assert round(np.max(Normalize()(noise))) == 1.0
    assert round(np.min(Normalize()(noise))) == -1.0
