from librosa.util import normalize


class Normalize(object):
    """Normalize
    Normalize for pytorch transform
    Returns:
        numpy.ndarray[numpy.float]
    """

    def __call__(self, data):
        normalized = normalize(data)
        return normalized
