from librosa.feature import melspectrogram
import numpy as np
import torch
from typing import Union, NewType, Optional, List, Any, Callable



class Melspectrogram(object):
    """Melspectrogram
    Melspectrogram for pytorch transform
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
        fmax (int, optional): max freq to process.
                              Default: `8000`
        abs (bool, optional): make output absolute value or not.
                              Default: `True`
    Returns:
        numpy.ndarray[numpy.float or numpy.complex] with shape (freq_reso, len(input signal)).
    """
    def __init__(self,
                *,
                n_mels: Optional[int]=None,
                hop_length: Optional[int]=None,
                win_length: Optional[int]=None,
                window: Optional[str]=None,
                fmax: Optional[int]=8000,
                abs: bool=True):
        self.n_mels: Optional[int] = n_mels if n_mels is not None else 128
        self.win_length: Optional[int] = win_length if win_length is not None else 2048
        self.hop_length: Optional[int] = hop_length if hop_length is not None else 2048//4
        self.window: Optional[str] = window if window is not None else 'hann'
        self.fmax: Optional[int] = fmax
        self.is_abs: bool = abs

    def __call__(self, data):
        s = melspectrogram(data,
                    n_mels=self.n_mels,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    fmax=self.fmax,
                    window=self.window)
        s = np.abs(s) if self.is_abs is True else s
        return s
