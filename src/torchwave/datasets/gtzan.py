from .dataset_folder import DatasetFolder
import numpy as np
from .utils import load, download_dataset
import torch
from typing import Union, NewType, Optional, List, Any, Callable


class GTZAN(DatasetFolder):
    def __init__(self,
                root: str, *,
                transform: Optional[Any]=None,
                one_hot: bool=False) -> None:
        self.url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
        self.root = download_dataset(self.url, root)
        super().__init__(self.root, transform=transform, one_hot=one_hot)
