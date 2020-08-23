import sys
sys.path.append('src')

from torchwave.transforms import Melspectrogram
import numpy as np
import pytest
import math

def test_init_Melspectrogram():
    assert Melspectrogram().n_mels == 128
    assert Melspectrogram().hop_length == 2048//4
    assert Melspectrogram().win_length == 2048
    assert Melspectrogram().window == 'hann'
    assert Melspectrogram().fmax == 8000
    assert Melspectrogram().is_abs == True

def test_process_Melspectrogram():
    noise = np.random.randn(2048)
    assert Melspectrogram()(noise).shape == (128, 4+1)
    assert Melspectrogram(n_mels=256)(noise).shape == (256, 4+1)
