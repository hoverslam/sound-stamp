import argparse
from pathlib import Path

import numpy as np

import torch
import librosa

from sound_stamp.tagger import MusicTagger
from sound_stamp.utils import to_log_mel_spectrogram, load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag an audio file using a trained model.")
    parser.add_argument("-i", "--input", type=str, metavar="",
                        help="File path to the audio file.")
    parser.add_argument("-m", "--model", type=str, default="music_tagger", metavar="",
                        help="File name of the trained model. Defaults to 'music_tagger'.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, metavar="",
                    help="Probability threshold for tag prediction. Defaults to 0.5.")
    args = parser.parse_args()
    audio_file = args.input
    prob_threshold = args.threshold
    model = args.model
    
    # Load configs
    configs = load_yaml(Path.cwd().joinpath("configs", "MusicTaggerFCN.yaml"))
    sample_rate = configs["audio"]["sample_rate"]
    fft_frame_size = configs["audio"]["fft_frame_size"]
    hop_size = configs["audio"]["hop_size"]
    num_mel_bands = configs["audio"]["num_mel_bands"]
    class_names = configs["datasets"]["MagnaTagATune"]["class_names"]
      
    # Initialize model and load state dict
    tagger = MusicTagger(configs)
    tagger.load(f"models/{model}.pt")

    # Load audio file and extract features
    waveform, _ = librosa.load(audio_file, sr=sample_rate)
    features = to_log_mel_spectrogram(waveform, sample_rate, fft_frame_size, hop_size, num_mel_bands)
    
    # Split longer audio files into multiple parts
    if features.shape[1] == 1366:
        features = features.unsqueeze(0)
    else: 
        chunks = torch.split(features, 1366, dim=1)        
        if len(chunks[-1][0]) < 1366:
            # If the last chunk is to small, drop it
            chunks = chunks[:-1]        
        features = torch.stack(chunks, dim=0)
    
    # Select tags based on the probability threshold
    class_names = np.array(class_names)
    probs = tagger.inference(features).cpu().numpy()
    tags = [tag for p in probs for tag in class_names[p > prob_threshold].tolist()]
    print(f"Tags: {list(set(tags))}")
