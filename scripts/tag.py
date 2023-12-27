import argparse

import numpy as np

import torch
import librosa

from sound_stamp.tagger import MusicTagger
from sound_stamp.utils import to_log_mel_spectrogram


# SETTINGS
SAMPLE_RATE = 12000
FFT_FRAME_SIZE = 512
NUM_MEL_BANDS = 96
HOP_SIZE = 256
AUDIO_LEN = 29.124   # seconds

# TODO: Put the class names somewhere else
class_names = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic',
               'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female',
               'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord',
               'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft',
               'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance',
               'female vocal', 'male voice', 'beats', 'harp', 'cello', 'no voice', 'weird',
               'country', 'female voice', 'metal', 'choral']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag an audio file using a trained model.")
    parser.add_argument("-i", "--input", type=str, metavar="",
                        help="File path to the audio file.")
    parser.add_argument("-m", "--model", type=str, default="music_tagger", metavar="",
                        help="File name of the trained model. Defaults to 'music_tagger'.")
    parser.add_argument("-t", "--threshold", type=float, default=0.25, metavar="",
                    help="Probability threshold for tag prediction. Defaults to 0.25.")
    args = parser.parse_args()
    audio_file = args.input
    prob_threshold = args.threshold
    model = args.model
      
    # Initialize model and load state dict
    tagger = MusicTagger(class_names)
    tagger.load(f"models/{model}.pt")

    # Load audio file and extract features
    waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    features = to_log_mel_spectrogram(waveform, SAMPLE_RATE, FFT_FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS)
    
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
