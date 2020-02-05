from torchwave.datasets.utils import load, download_file, safe_path, extract
import numpy as np
import scipy.io.wavfile as siw
import pytest
import os
import shutil


def test_load_npy():
    np.save('./test.npy', np.random.randn(10))
    assert load('./test.npy').shape == (10,)
    assert load('./test.npy', duration=5).shape == (5,)
    assert load('./test.npy', offset=3).shape == (7,)
    assert load('./test.npy', offset=3, duration=5).shape == (5,)

    with pytest.raises(TypeError):
        load('./test.npy', duration=0.1)
        load('./test.npy', offset=0.3)

    with pytest.raises(IndexError):
        load('./test.npy', offset=11)

    with pytest.warns(UserWarning):
        load('./test.npy', offset=3, duration=8)

def test_load_audio():
    # gen wav
    noise = np.random.randn(22050)
    int16amp = 32768
    noise = np.array([noise * int16amp], dtype = "int16")[0]
    noise =np.array(np.c_[noise, noise])
    siw.write('./test.wav', 22050, noise)

    assert load('./test.wav').shape == (22050,)
    assert load('./test.wav', duration=0.5).shape == (22050//2,)
    assert load('./test.wav', sr=44100).shape == (44100,)
    assert load('./test.wav', offset=0.2, duration=0.5).shape == (22050//2,)
    assert load('./test.wav', mono=False).shape == (2, 22050)

    # with pytest.raises(ValueError):
    print(load('./test.wav', offset=0))

def test_download_file():
    filename = download_file('http://nginx.org/download/nginx-0.1.0.tar.gz')
    assert filename == safe_path('./nginx-0.1.0.tar.gz')
    os.remove(filename)

    os.makedirs('./data')
    filename = download_file('http://nginx.org/download/nginx-0.1.0.tar.gz', './data')
    assert filename == safe_path('./data/nginx-0.1.0.tar.gz')
    os.remove(filename)
    os.rmdir('./data')

def test_extract():
    os.makedirs('./data')
    filename = download_file('http://nginx.org/download/nginx-0.1.0.tar.gz', './data')
    assert extract(filename) == safe_path('./data/nginx-0.1.0')
    shutil.rmtree('./data')
