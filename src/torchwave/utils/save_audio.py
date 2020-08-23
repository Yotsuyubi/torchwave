import torch
import soundfile as sf
from ..transforms import Normalize


def save_audio(tensor, fp, *, normalize=False, fs=22050):
    tensor = tensor.cpu().numpy().flatten()
    tensor = tensor if normalize is False else Normalize()(tensor)
    sf.write('{}'.format(fp), tensor, fs, subtype='PCM_24')
