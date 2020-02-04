from torchwave.transforms import RandomCrop
import numpy as np
import pytest
import math

def test_init_RandomCrop():
    with pytest.raises(TypeError):
        RandomCrop()


def test_process_RandomCrop():
    noise = np.random.randn(2048)
    assert RandomCrop(length=512)(noise).shape == (512,)

    with pytest.raises(ValueError):
        RandomCrop(length=-512)(noise)
        RandomCrop(length=2048)(noise)
