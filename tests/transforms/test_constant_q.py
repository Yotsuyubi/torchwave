import sys
sys.path.append('src')

from torchwave.transforms.constant_q import ConstantQ
import numpy as np
import pytest
import math

def test_init_ConstantQ():
    assert ConstantQ().sampling_rate == 22050
    assert ConstantQ().hop_length == 512
    assert ConstantQ().n_bins == 84
    assert ConstantQ().bins_per_octave == 12
    assert ConstantQ().window == 'hann'
    assert ConstantQ().is_abs == True

def test_process_ConstantQ():
    noise = np.random.randn(22050)
    assert ConstantQ()(noise).shape == (84, math.ceil(22050/512))
    assert ConstantQ(n_bins=10)(noise).shape == (10, math.ceil(22050/512))
