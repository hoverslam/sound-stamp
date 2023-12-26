import argparse

import numpy as np
import librosa

from sound_stamp.datasets import MagnaSet
from sound_stamp.tagger import MusicTagger
from sound_stamp.utils import to_log_mel_spectrogram


# SETTINGS
SAMPLE_RATE = 12000
FFT_FRAME_SIZE = 512
NUM_MEL_BANDS = 96
HOP_SIZE = 256
AUDIO_LEN = 29.124   # seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag an audio file using a trained model.")
    parser.add_argument("--input", "-i", type=str,
                        help="File path to the audio file.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Probability threshold for tag prediction. Defaults to 0.5.")
    parser.add_argument("--file_name", "-f", type=str, default="music_tagger.pt",
                        help="File name of the trained model. Defaults to 'music_tagger.pt'.")
    args = parser.parse_args()
    audio_file = args.input
    prob_threshold = args.threshold
    file_name = args.file_name
    
    # Load data set => needed for the class names
    # TODO: Find another way to load class names
    dataset = MagnaSet()
   
    # Initialize model and load state dict
    tagger = MusicTagger(dataset.class_names)
    tagger.load(f"models/{file_name}")

    # Load audio file and extract features
    # TODO: Split longer audio into parts and tag all off them
    waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE, duration=AUDIO_LEN)
    features = to_log_mel_spectrogram(waveform, SAMPLE_RATE, FFT_FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS)
    features = features.unsqueeze(0)
    
    # Select tags based on the probability threshold
    class_names = np.array(dataset.class_names)
    predicted_tags = tagger.inference(features).cpu().numpy()
    print(f"Tags: {class_names[predicted_tags > prob_threshold]}")
