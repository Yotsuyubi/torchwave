from torch.utils.data import Dataset
from torchwave.datasets.utils import match_length
import numpy as np
import torch


class Remix(Dataset):
    def __init__(self, datasets, *, transform=None, shuffle=True, label_index=-1):
        super().__init__()
        self.datasets = datasets
        self.transform = transform
        self.shuffle = shuffle
        self.label_index = label_index

    def __len__(self):
        lengths = [d.__len__() for d in self.datasets]
        return min(lengths)

    def __getitem__(self, idx):
        signals = []
        for dataset in self.datasets:
            if self.shuffle is True:
                np.random.shuffle(dataset.filenames)
            signal, _ = dataset[idx]
            signal = signal[0]
            if len(signals) > 0:
                signal_before = signals[-1]
                signal_before, signal = match_length(signal_before, signal)
                signals[-1] = signal_before
            signals.append(signal)
        signal = np.sum(signals, axis=0)
        signal = self.transform(signal) if self.transform is not None else signal
        label = self.transform(signals[self.label_index]) if self.transform is not None else signals[self.label_index]
        return signal, label
