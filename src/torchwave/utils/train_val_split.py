import torch
import random
import math
from ..datasets import DatasetFolder


def train_val_split(dataset, val, *, seed=0):
    
    random.seed(seed)

    val_length = math.floor(len(dataset) * val)
    train_length = len(dataset) - val_length
    train_filenames = random.sample(dataset.filenames, train_length)
    val_filenames = [filename for filename in dataset.filenames if filename not in train_filenames]

    root = dataset.root
    transform = dataset.transform
    one_hot = dataset.one_hot
    label = dataset.label

    train_dataset = DatasetFolder(root, transform=transform, one_hot=one_hot, label=label)
    train_dataset.filenames = train_filenames
    val_dataset = DatasetFolder(root, transform=transform, one_hot=one_hot, label=label)
    val_dataset.filenames = val_filenames

    return train_dataset, val_dataset
