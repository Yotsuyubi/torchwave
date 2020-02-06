from torchwave.datasets import GTZAN

dataset = GTZAN('./data')
sig, label = dataset[0]
print(sig, dataset.index_to_class(label))
