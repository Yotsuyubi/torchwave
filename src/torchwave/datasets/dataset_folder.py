from torch.utils.data import Dataset
import glob
import re
import numpy as np
from typing import Union, NewType, Optional, List, Any, Callable
from .utils import load
import torch

Path = str
Transform = Any
Signal = Any


class DatasetFolder(Dataset):
    """Load npy or audio files from folder
    Args:
        root (str): root path of files.
        transform (torch transform, optional): Default `None`
        one_hot (bool, optional): return label as one-hot array or not.
                                  Default `false`
    """
    def __init__(self,
                root: Path, *,
                transform: Optional[Transform]=None,
                one_hot: bool=False,
                label: bool=True) -> None:
        super(DatasetFolder, self).__init__()

        self.root: Path = root
        self.classes: List[str] = self._filename_to_classes()
        self.is_one_hot: bool = one_hot
        self.filenames: List[Path] = glob.glob(self.root+'/**/*')
        self.transform: Optional[Transform] = transform
        self.label = label

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename: Path = self.filenames[idx]
        data: Signal = load(filename)

        if len(data.shape) < 2:
            # expand channel for time-series data
            data = np.expand_dims(data, 0)

        if self.transform is not None: # apply transform for each channels
            datas: List[Signal] = []
            for d in data:
                transformed_signal: Signal = self.transform(d)
                datas.append(transformed_signal)
            data = np.concatenate(datas, 0)

        if self.label is True:
            label: str = self._get_class(filename)
            y = self.classes.index(label)
            y = torch.tensor(y).long()
            if self.is_one_hot is True:
                y = np.eye(len(self.classes))[self.classes.index(label)]
                y = torch.tensor(y).float()
            return data, y
        else:
            return data, torch.tensor([0]).long()

    def index_to_class(self, index: int) -> str:
        """get class name of index given
        Args:
            index (int): index of classes
        Returns:
            class (str): class name
        """
        return self.classes[index]

    def _filename_to_classes(self) -> List[str]:
        """get classes from filenames
        """
        l = glob.glob(self.root+'/**/')
        labels = [dir.replace(self.root, '') for dir in l]
        labels = [dir.replace('/', '') for dir in labels]
        labels = [re.sub(r'\.[a-z0-9]+', '', dir) for dir in labels]
        return labels

    def _get_class(self, filename: Path) -> str:
        dir_fn = filename.replace(self.root, '')
        dir = re.match(r'/[a-z0-9]+/', dir_fn).group()
        dir = dir.replace('/', '')
        return dir
