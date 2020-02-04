import numpy as np
import torch

def acc(y_hat, y):
    y_hat = torch.tensor(y_hat).cpu().numpy()
    y = torch.tensor(y).cpu().numpy()

    if len(y.shape) == 2:   # if y is one-hot vector
        y = np.argmax(y, axis=1) # to index

    y_hat = np.argmax(y_hat, axis=1)
    score = (y_hat == y).mean()

    return score
