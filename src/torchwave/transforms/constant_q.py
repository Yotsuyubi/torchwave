from librosa.core import cqt
import numpy as np
import torch
from typing import Union, NewType, Optional, List, Any, Callable



class ConstantQ(object):
    """Constant-Q
    Constant-Q for pytorch transform
    Args:
        sampling_rate (int, optional): sampling rate of input signal.
                                       `None` to be 22050.
                                       Default: `None`
        hop_length (int, optional): hop length. `None` to be 512.
                                    Default: `None`
        n_bins (int, optional): Number of frequency bins, starting at fmin. `None` to be 84.
                                Default: `None`
        bins_per_octave (int, optional): Number of bins per octave. `None` to be 12
                                         Default: `12`
        window (str, optional): window function. `None` to be `hann`
                                Default: `None`
        abs (bool, optional): make output absolute value or not.
                              Default: `True`
    Returns:
        numpy.ndarray[numpy.float or numpy.complex] with shape (freq_reso, len(input signal)).
    """
    def __init__(self,
                *,
                sampling_rate: Optional[int]=None,
                hop_length: Optional[int]=None,
                n_bins: Optional[int]=None,
                bins_per_octave: Optional[int]=None,
                window: Optional[str]=None,
                abs: bool=True):
        self.sampling_rate: Optional[int] = sampling_rate if sampling_rate is not None else 22050
        self.hop_length: Optional[int] = hop_length if hop_length is not None else 512
        self.n_bins: Optional[int] = n_bins if n_bins is not None else 84
        self.bins_per_octave: Optional[int] = bins_per_octave if bins_per_octave is not None else 12
        self.window: Optional[str] = window if window is not None else 'hann'
        self.is_abs: bool = abs

    def __call__(self, data):
        constantq = cqt(data,
                        sr=self.sampling_rate,
                        hop_length=self.hop_length,
                        n_bins=self.n_bins,
                        bins_per_octave=self.bins_per_octave,
                        window=self.window)
        constantq = np.abs(constantq) if self.is_abs is True else constantq
        return constantq
