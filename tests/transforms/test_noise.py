import sys
sys.path.append('src')

from torchwave.transforms import Noise
import numpy as np
import pytest
import math

def test_init_Noise():
    assert Noise().power == 0.1

def test_process_Noise():
    noise = np.random.randn(2048)
    assert Noise()(noise).shape == (2048,)
    assert Noise(power=0.2)(noise).shape == (2048,)
