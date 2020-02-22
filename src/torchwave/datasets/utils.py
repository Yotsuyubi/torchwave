import numpy as np
import librosa
import os
from warnings import warn
from typing import Union, NewType, Optional
import requests
import tarfile
import pathlib
from tqdm import tqdm

Path = NewType('Path', str)


def load(filename: str, *,
            sr: int=22050,
            mono: bool=True,
            duration: Union[int, float, None]=None,
            offset: Union[int, float]=0):
    """Load audio file either npy file.
    Args:
        filename (str): filename to load.
        sr (int, optional): sampling rate of audio sample.
        mono (bool, optional): audio sample will be either mono or stereo.
                               Default: `True`.
        duration (positive int or float, optional): for numpy file: length of samples. duration must be type of 'int'
                                           for audio file: duration of audio samples.
                                           Default: None
        offset (int or float, optional): for numpy file: length of samples. offset must be type of 'int'
                                           for audio file: offset of audio samples.
                                           Default: 0
    Returns:
        signal (numpy.array)
    """

    signal: np.ndarray[np.float64] = np.array([.0])
    filename = safe_path(filename)
    _, ext = os.path.splitext(filename) # get file extention

    try:
        if duration is not None and duration < 0:
            raise ValueError('`duration` must be positive int or float.')

        if ext == '.npy': # if file which loading is numpy format.
            if (duration is not None and type(duration) is not int) or type(offset) is not int:
                raise TypeError('`duration` and `offset` must be int.')
            d: Union[int, None] = int(duration) if type(duration) is int else None
            o: int = int(offset)
            signal = _load_npy(filename, duration=d, offset=o)

        else: # if file which loading is audio format.
            signal, _ = librosa.load(
                filename,
                sr=sr,
                mono=mono,
                duration=duration,
                offset=offset
            )
    except ValueError:
        warn('File opening error for: {}'.format(filename), UserWarning)
    except (IOError, FileNotFoundError):
        warn('File does not exist: {}'.format(filename), UserWarning)

    return signal

def download_file(url, path='./'):
    root_path = safe_path(path)
    local_filename = url.split('/')[-1]
    local_filename = safe_path(root_path+'/'+local_filename)
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    file_size = int(requests.head(url).headers["content-length"])
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)
    print('Download from {}...'.format(url))
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
                pbar.update(len(chunk))
        pbar.close()
    return local_filename

def extract(filename):
    dirname = os.path.dirname(filename)
    basename = basename_without_ext(filename)
    with tarfile.open(filename, 'r:*') as tar:
        tar.extractall(dirname)
        os.remove(filename)
    return dirname+'/'+basename

def basename_without_ext(path):
    basename_without_ext = pathlib.Path(path).resolve().stem
    basename_without_ext = pathlib.Path(basename_without_ext).resolve().stem
    return basename_without_ext

def download_dataset(url, path='./'):
    dirname = safe_path(path)+'/'+basename_without_ext(url)
    if os.path.exists(dirname) is False:
        local_filename = download_file(url, path)
        dirname = extract(local_filename)
    return dirname

def _load_npy(filename: Path,
                duration: Optional[int]=None,
                offset: int=0):
    duration = offset+duration if duration is not None else None
    signal: np.ndarray[np.float64] = np.array(np.load(filename), dtype=np.float64)

    # error / warn handling
    if len(signal) < offset:
        raise IndexError('`offset` out of range.')
    if duration is not None and duration > len(signal):
        warn('`duration` out of range. it fixed to None.', UserWarning)
        duration = None

    signal = signal[offset:duration]
    return signal

def safe_path(path: str) -> Path:
    """Ensure the path is absolute and doesn't include `..` or `~`.
       site: 'https://github.com/audeering/audtorch/blob/0.4.1/audtorch/datasets/utils.py#L359'
    Args:
        path (str): absolute or relative path
    Returns:
        str: absolute path
    """

    return os.path.abspath(os.path.expanduser(path))

def match_length(sig_a, sig_b):
    a_len = len(sig_a)
    b_len = len(sig_b)
    if a_len < b_len:
        sig_b = sig_b[:a_len]
    else:
        sig_a = sig_a[:b_len]
    return sig_a, sig_b
