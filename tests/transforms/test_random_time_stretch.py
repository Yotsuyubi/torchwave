import sys
sys.path.append('src')

from torchwave.transforms import RandomTimeStretch
import numpy as np
import pytest
from librosa.util.exceptions import ParameterError

def test_RandomTimeStretch():
    noise = np.random.randn(512)
    assert RandomTimeStretch()(noise).shape == (512,)
    assert RandomTimeStretch(rate=2)(noise).shape == (512,)
    assert RandomTimeStretch(rate=[0.5, 1.5])(noise).shape == (512,)

    with pytest.raises(ValueError):
        RandomTimeStretch(rate=[0.5, 1.5, 1])(noise)
