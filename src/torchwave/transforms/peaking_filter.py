import scipy.signal as signal
import numpy as np
from .normalize import Normalize

class PeakingFilter(object):
    def __init__(self,
                f: float,
                q: float,
                gain: float,
                *,
                fs: int=22050,
                normalize: bool=True):
        self.f: float = f
        self.q: float = q
        self.gain: float = gain
        self.fs: int = fs
        self.normalize: bool = True
        self.filter = self._peaking_eq(self.q, self.gain, self.f, self.fs)

    def __call__(self, data):
        filtered = self._apply(self.filter[0], self.filter[1], data)
        if self.normalize is True:
            filtered = Normalize()(filtered)
        return filtered

    def _peaking_eq(self, q: float, gain: float, f: float, fs: int):
        if f > fs/2 or f < 0:
            raise ValueError('`f` should be less than `fs/2` and potisive float.')
        if int(q) < 1 or q > 24:
            raise ValueError('`q` should be less than `24` and positive float but not `0`.')
        if gain < -24 or gain > 24:
            raise ValueError('`gain` should be less than `24` and grater than `-24`.')

        A: float = 10 ** (gain / 40.0)
        w0: float = 2.0 * np.pi * f / fs
        alpha: float = np.sin(w0) / (2.0 * q)

        b = [(1.0 + alpha * A), (-2.0 * np.cos(w0)), (1.0 - alpha * A)]
        a = [(1.0 + alpha / A), (-2.0 * np.cos(w0)), (1.0 - alpha / A)]

        return np.array(b), np.array(a)

    def _apply(self, b, a, x):
        y = []
        in1 = 0
        in2 = 0
        out1 = 0
        out2 = 0
        for input in x:
            output = b[0]/a[0]*input + b[1]/a[0]*in1 + b[2]/a[0]*in2 \
                                     - a[1]/a[0]*out1 - a[2]/a[0]*out2
            y.append(output)

            in2 = in1
            in1 = input
            out2 = out1
            out1 = output
        return np.array(y)
