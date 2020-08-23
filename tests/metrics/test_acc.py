import sys
sys.path.append('src')

from torchwave.metrics import acc
import numpy as np
import torch

def test_acc():
    y = np.array([[1, 0, 0, 0]])
    y_hat = np.array([[1, 0, 0, 0]])
    assert acc(y_hat, y) == 1
    y = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    y_hat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert acc(y_hat, y) == 0.5
    y = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])
    y_hat = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert acc(y_hat, y) == 0.5
