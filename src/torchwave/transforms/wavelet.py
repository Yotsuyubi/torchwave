import numpy as np
import torch
import scipy.signal as sig
import scipy
from typing import Union, NewType, Optional, List, Any, Callable



class Wavelet(object):
    """Mother Wavelet
    Mother Wavelet for pytorch transform
    Args:
        sampling_rate (int, optional): sampling rate of input signal.
                                       `None` to be same as inputed data.
                                       Default: `None`
        freq_reso (int, optional): resolution of frequency domain. `None` to be 100.
                                   Default: `None`
        w (int, optional): omega of scipy.signal.cwt. `None` to be `6`.
                           Default: `None`
        abs (bool, optional): make output absolute value or not.
                              Default: `True`
    Returns:
        numpy.ndarray[np.float or np.complex] with shape (freq_reso, len(input signal)).
    """
    def __init__(self,
                *,
                sampling_rate: Optional[int]=None,
                freq_reso: Optional[int]=None,
                w: Optional[int]=None,
                abs: bool=True):
        self.sampling_rate: Optional[int] = sampling_rate
        self.freq_reso: int = freq_reso if freq_reso is not None else 100
        self.w: int = w if w is not None else 6
        self.is_abs: bool = abs

    def __call__(self, data):
        """
        site: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.morlet2.html#scipy.signal.morlet2
        """
        fs: int = self.sampling_rate if self.sampling_rate is not None else len(data)
        freq: np.ndarray[np.float64] = np.linspace(1, fs/2, self.freq_reso)
        widths: np.ndarray[np.float64] = self.w*fs / (2*freq*scipy.pi)
        cwtmatr = sig.cwt(data, sig.morlet2, widths, w=self.w)
        cwtmatr = np.abs(cwtmatr) if self.is_abs is True else cwtmatr
        return cwtmatr
