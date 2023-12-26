import numpy as np

import torch 
import torchaudio


def to_log_mel_spectrogram(waveform: np.ndarray, sample_rate: int, 
                           n_fft: int, hop_length: int, n_mels: int) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    mel_spec = mel_transform(torch.from_numpy(waveform))
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return log_mel_spec