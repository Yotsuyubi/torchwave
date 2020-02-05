from torchwave.transforms import TimeStretch, Repeat
import numpy as np
import pytest
import math


def test_process_Repeat():
    noise = np.random.randn(512)
    assert Repeat(TimeStretch(0.5), 2)(noise).shape == (512*4,)
