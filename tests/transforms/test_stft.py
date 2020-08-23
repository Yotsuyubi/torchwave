import sys
sys.path.append('src')

from torchwave.transforms.stft import STFT
import numpy as np
import pytest
import math

def test_init_STFT():
    assert STFT().n_fft == 2048
    assert STFT().hop_length == 2048//4
    assert STFT().win_length == 2048
    assert STFT().window == 'hann'
    assert STFT().is_abs == True

def test_process_ConstantQ():
    noise = np.random.randn(2048)
    assert STFT()(noise).shape == (2048//2+1, 4+1)
