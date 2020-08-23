import sys
sys.path.append('src')

from torchwave.transforms.wavelet import Wavelet
import numpy as np
import pytest

def test_wavelet():
    noise = np.random.randn(512)
    assert Wavelet().sampling_rate == None
    assert Wavelet().freq_reso == 100
    assert Wavelet()(noise).shape == (100,512)
    assert Wavelet(sampling_rate=128)(noise).shape == (100,512)
    assert Wavelet(freq_reso=80)(noise).shape == (80,512)
    assert Wavelet()(noise).dtype == 'float64'
    assert Wavelet(abs=False)(noise).dtype == 'complex128'

    with pytest.raises(TypeError):
        Wavelet(freq_reso=0.1, sampling_rate=512)(noise)
        Wavelet(freq_reso=80, sampling_rate=0.512)(noise)
        Wavelet(freq_reso=80, sampling_rate=512, w=0.1)(noise)
