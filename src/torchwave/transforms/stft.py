from librosa.core import stft
import numpy as np
import torch
from typing import Union, NewType, Optional, List, Any, Callable



class STFT(object):
    """STFT
    STFT for pytorch transform
    Args:
        n_fft (int, optional): sampling rate of input signal.
                                       `None` to be 22050.
                                       Default: `None`
        hop_length (int, optional): hop length. `None` to be 512.
                                    Default: `None`
        win_length (int, optional): Number of frequency bins, starting at fmin. `None` to be 84.
                                Default: `None`
        window (str, optional): window function. `None` to be `hann`
                                Default: `None`
        abs (bool, optional): make output absolute value or not.
                              Default: `True`
    Returns:
        numpy.ndarray[numpy.float or numpy.complex] with shape (freq_reso, len(input signal)).
    """
    def __init__(self,
                *,
                n_fft: Optional[int]=None,
                hop_length: Optional[int]=None,
                win_length: Optional[int]=None,
                window: Optional[str]=None,
                abs: bool=True):
        self.n_fft: Optional[int] = n_fft if n_fft is not None else 2048
        self.win_length: Optional[int] = win_length if win_length is not None else self.n_fft
        self.hop_length: Optional[int] = hop_length if hop_length is not None else self.n_fft//4
        self.window: Optional[str] = window if window is not None else 'hann'
        self.is_abs: bool = abs

    def __call__(self, data):
        s = stft(data,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    window=self.window)
        s = np.abs(s) if self.is_abs is True else s
        return s
