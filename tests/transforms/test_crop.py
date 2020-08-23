import sys
sys.path.append('src')

from torchwave.transforms.crop import Crop
import numpy as np
import pytest
import math

def test_init_Crop():
    assert Crop(length=512).start == 0
    with pytest.raises(TypeError):
        Crop()


def test_process_Crop():
    noise = np.random.randn(2048)
    assert Crop(length=512)(noise).shape == (512,)
    assert Crop(length=512, start=100)(noise).shape == (512,)
    assert Crop(length=512, start=2048-511)(noise).shape == (511,)
    assert Crop(length=2049)(noise).shape == (2048,)

    with pytest.raises(IndexError):
        Crop(length=512, start=2049)(noise)

    with pytest.raises(ValueError):
        Crop(length=-512, start=0)(noise)
        Crop(length=512, start=-1)(noise)
