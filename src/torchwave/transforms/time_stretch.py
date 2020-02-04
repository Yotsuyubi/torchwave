from librosa.effects import time_stretch


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
        time_stretched = time_stretch(data, self.rate)
        return time_stretched
