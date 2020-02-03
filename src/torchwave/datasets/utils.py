import numpy as np
import librosa
import os
import warnings


def load(filename, *, sr=22050, mono=True, duration=None, offset=0):
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

    signal = np.array([])
    filename = safe_path(filename)
    _, ext = os.path.splitext(filename) # get file extention

    try:
        if duration is not None and duration < 0:
            raise ValueError('`duration` must be positive int or float.')

        if ext == '.npy': # if file which loading is numpy format.
            signal = _load_npy(filename, duration, offset)

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


def _load_npy(filename, duration, offset):
    duration = offset+duration if duration is not None else None
    signal = np.load(filename)

    # error / warn handling
    if len(signal) < offset:
        raise IndexError('`offset` out of range.')
    if duration is not None and duration > len(signal):
        warnings.warn('`duration` out of range. it fixed to None.', UserWarning)
        duration = None

    signal = signal[offset:duration]
    return signal

def safe_path(path):
    """Ensure the path is absolute and doesn't include `..` or `~`.
       site: 'https://github.com/audeering/audtorch/blob/0.4.1/audtorch/datasets/utils.py#L359'
    Args:
        path (str): absolute or relative path
    Returns:
        str: absolute path
    """

    return os.path.abspath(os.path.expanduser(path))
