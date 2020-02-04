from torchwave.transforms import Crop, RandomApply
import numpy as np
import pytest
import math


def test_process_Crop():
    noise = np.random.randn(2048)
    crop = Crop(length=512)
    assert RandomApply(p=0, transform=crop)(noise).shape == (2048,)
    assert RandomApply(p=1, transform=crop)(noise).shape == (512,)

    with pytest.raises(ValueError):
        RandomApply(p=-1, transform=crop)(noise)
        RandomApply(p=2, transform=crop)(noise)
