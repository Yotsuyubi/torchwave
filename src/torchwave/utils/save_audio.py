import torch
from librosa.output import write_wav
from ..transforms import Normalize


def save_audio(tensor, fp, *, normalize=False, fs=22050):
    tensor = tensor.cpu().numpy().flatten()
    tensor = tensor if normalize is False else Normalize()(tensor)
    write_wav('{}'.format(fp), tensor, fs)
