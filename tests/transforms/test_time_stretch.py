from torchwave.transforms import TimeStretch
import numpy as np
import pytest
from librosa.util.exceptions import ParameterError

def test_TimeStretch():
    noise = np.random.randn(512)
    assert TimeStretch(0.5)(noise).shape == (512,)
    assert TimeStretch(2)(noise).shape == (512,)

    with pytest.raises(ParameterError):
        TimeStretch(0)(noise)
