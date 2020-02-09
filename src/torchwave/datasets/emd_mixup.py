from torch.utils.data import Dataset
from torchwave.datasets.utils import load
import numpy as np
import torch
from PyEMD import EMD


class EMDMixup(Dataset):
    def __init__(self, dataset, *, p=1.0):
        super().__init__()
        self.dataset = dataset
        self.p = p

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        filename = self.dataset.filenames[idx]
        data = load(filename)
        label = self.dataset._get_class(filename)
        data = self._emd(data, label) if np.random.rand() < self.p else data

        if len(data.shape) < 2:
            # expand channel for time-series data
            data = np.expand_dims(data, 0)

        if self.dataset.transform is not None: # apply transform for each channels
            datas= []
            for d in data:
                transformed_signal= self.dataset.transform(d)
                datas.append(transformed_signal)
            data = np.concatenate(datas, 0)

        y = self.dataset.classes.index(label)
        y = torch.tensor(y).long()
        if self.dataset.is_one_hot is True:
            y = np.eye(len(self.dataset.classes))[self.dataset.classes.index(label)]
            y = torch.tensor(y).float()

        return data, y

    def _emd(self, data, label):
        f_0 = self._random_pick(label)
        d = load(f_0)

        if len(d) < len(data):
            data = data[:len(d)]
        else:
            d = d[:len(data)]

        emd = EMD()
        emd.FIXE = 5
        imf_0 = emd(data, max_imf=5)
        imf_1 = emd(d, max_imf=5)

        imf = [*imf_0[:3], *imf_1[3:]]
        data = np.sum(imf, 0)
        return data

    def _random_pick(self, label):
        filenames = self.dataset.filenames
        np.random.shuffle(filenames)
        for f in filenames:
            l = self.dataset._get_class(f)
            if label == l:
                return f
        raise ValueError()
