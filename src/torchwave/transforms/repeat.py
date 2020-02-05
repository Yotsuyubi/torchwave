class Repeat(object):
    """Repeat
    Repeat transform
    Args:
        transform (callable): transform
        n (int): num for repetation
    Returns:
        numpy.ndarray[numpy.float]
    """

    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, data):
        x = data
        for i in range(self.n):
            x = self.transform(x)
        return x
