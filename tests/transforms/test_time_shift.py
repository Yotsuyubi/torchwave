import sys
sys.path.append('src')

from torchwave.transforms import TimeShift
import numpy as np
import pytest
import math

def test_init_TimeShift():
    assert TimeShift().shift == 0


def test_process_TimeShift():
    noise = np.random.randn(2048)
    assert TimeShift()(noise).shape == (2048,)
    assert TimeShift(shift=-50)(noise).shape == (2048,)
    assert TimeShift(shift=50)(noise).shape == (2048,)

    with pytest.raises(TypeError):
        TimeShift(shift=0.1)(noise)
