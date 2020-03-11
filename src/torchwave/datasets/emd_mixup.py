from torch.utils.data import Dataset
from torchwave.datasets.utils import load, match_length
import numpy as np
import torch
from PyEMD import EMD


class EMDMixup(Dataset):
    def __init__(self, dataset, *, p=1.0, pretransform=None, subnet=False):
        super().__init__()
        self.dataset = dataset
        self.p = p
        self.pretransform = pretransform
        self.subnet = subnet

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        filename = self.dataset.filenames[idx] if self.subnet is False else self.dataset.dataset.filenames[idx]
        data = load(filename)
        if self.pretransform is not None:
            data = self.pretransform(data)
        label = self.dataset._get_class(filename) if self.subnet is False else self.dataset.dataset._get_class(filename)
        data = self._emd(data, label) if np.random.rand() < self.p else data

        if len(data.shape) < 2:
            # expand channel for time-series data
            data = np.expand_dims(data, 0)

        transform = self.dataset.transform if self.subnet is False else self.dataset.dataset.transform
        if transform is not None: # apply transform for each channels
            datas= []
            for d in data:
                transformed_signal= transform(d)
                datas.append(transformed_signal)
            data = np.concatenate(datas, 0)

        classes = self.dataset.classes if self.subnet is False else self.dataset.dataset.classes

        y = classes.index(label)
        y = torch.tensor(y).long()

        is_one_hot = self.dataset.is_one_hot if self.subnet is False else self.dataset.dataset.is_one_hot
        if is_one_hot is True:
            y = np.eye(len(classes))[classes.index(label)]
            y = torch.tensor(y).float()

        return data, y

    def _emd(self, data, label):
        f_0 = self._random_pick(label)
        d = load(f_0)

        if self.pretransform is not None:
            d = self.pretransform(d)

        data, d = match_length(data, d)

        emd = EMD()
        emd.FIXE = 5
        imf_0 = emd(data, max_imf=5)
        imf_1 = emd(d, max_imf=5)

        imf = [*imf_0[:3], *imf_1[3:]]
        data = np.sum(imf, 0)
        return data

    def _random_pick(self, label):
        if self.subnet is False:
            filenames = self.dataset.filenames[:] # deepcopy
            np.random.shuffle(filenames)
            for f in filenames:
                l = self.dataset._get_class(f)
                if label == l:
                    return f
            raise ValueError()
        else:
            indices = self.dataset.indices[:]
            np.random.shuffle(indices)
            for i in indices:
                filename = self.dataset.dataset.filenames[i]
                l = self.dataset.dataset._get_class(filename)
                if label == l:
                    return filename
            raise ValueError()
