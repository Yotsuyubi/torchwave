from librosa.effects import time_stretch
import numpy as np


class TimeStretch(object):
    """TimeStretch
    TimeStretch for pytorch transform
    Args:
        rate (float) > 0: Stretch factor.
                      If rate > 1, then the signal is sped up.
                      If rate < 1, then the signal is slowed down.
    Returns:
        numpy.ndarray[numpy.float]
    """

    def __init__(self, rate):
        self.rate: float = rate

    def __call__(self, data):
        original_len = len(data)
        time_stretched = time_stretch(data, self.rate)
        if len(time_stretched) < original_len:
            time_stretched = np.pad(time_stretched, (0, max(0, original_len-len(time_stretched))), "constant")
        else:
            time_stretched = time_stretched[:original_len]
        return time_stretched
