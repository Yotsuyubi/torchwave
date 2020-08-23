import sys
sys.path.append('src')

from torchwave.utils import save_audio
from torchwave.datasets.utils import load
import torch
import numpy as np
import pytest
import os


def test_process_save_audio():
    noise = torch.randn(44100)
    save_audio(noise, 'test.wav')
    noise = load('test.wav')
    assert noise.shape == (44100, )
    noise = torch.randn(5, 44100)
    save_audio(noise, 'test.wav')
    noise = load('test.wav')
    assert noise.shape == (44100*5, )

    os.remove('./test.wav')
