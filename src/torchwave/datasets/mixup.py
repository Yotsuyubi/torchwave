from torch.utils.data import Dataset

import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
import torch
from torch.utils.data import random_split

Path = str
Transform = Any
Signal = Any


class Mixup(Dataset):
    """Mixup
    Args:
        dataset (dataset)
    """
    def __init__(self,
                dataset: Any) -> None:
        super(Mixup, self).__init__()

        self.org_dataset = dataset
        self.org_dataset.one_hot = True
        self.dataset_len = len(self.org_dataset)//2
        self.A, self.B = random_split(self.org_dataset, [self.dataset_len, self.dataset_len])

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, idx: int):
        sig_A, label_A = self.A[idx]
        sig_B, label_B = self.B[idx]
        alpha = np.random.rand()
        data = alpha*sig_A+(1-alpha)*sig_B
        label = alpha*label_A+(1-alpha)*label_B
        return data, label.astype(np.float32)
