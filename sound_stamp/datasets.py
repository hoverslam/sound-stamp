import warnings
import random
import requests
import zipfile
from pathlib import Path

import pandas as pd

import librosa
import torch
from torch.utils.data import Dataset, Subset

from sound_stamp.utils import to_log_mel_spectrogram


SAMPLE_RATE = 12000
FFT_FRAME_SIZE = 512
NUM_MEL_BANDS = 96
HOP_SIZE = 256
        

class MagnaSet(Dataset):
    def __init__(self, num_classes: int = 50) -> None:
        super().__init__()

        self.path = Path.cwd().joinpath("data", "magna")
        self.num_classes = num_classes

        self._download()
        self._create_targets()

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                audio_path = self.path.joinpath("audio", self.audio_files[idx])
                waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
                features = to_log_mel_spectrogram(waveform, SAMPLE_RATE, FFT_FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS)
                targets = self.targets[idx].type(torch.float32)
            except:
                features = torch.zeros((96, 1366), dtype=torch.float32)
                targets = torch.zeros(self.num_classes, dtype=torch.float32)     
        
        return features, targets

    def _download(self) -> None:
        url = "https://mirg.city.ac.uk/datasets/magnatagatune"
        files = [
            "mp3.zip.001",
            "mp3.zip.002",
            "mp3.zip.003",
            "annotations_final.csv",
            "clip_info_final.csv",
        ]

        print("Downloading MagnaTagATune ... ")
        if not self.path.exists():
            self.path.mkdir(parents=True)

            # Download data
            for file in files:
                response = requests.get(f"{url}/{file}")
                if response.status_code == 200:
                    with open(self.path.joinpath(file), "wb") as f:
                        f.write(response.content)
                    print(f"File downloaded to '{self.path.joinpath(file)}'.")
                else:
                    print(
                        f"Failed to download '{file}'. Status code: {response.status_code}")

            # Concatenate the multi-part ZIP files into one ZIP file
            print("Unzipping audio files ... ")
            with open(self.path.joinpath("mp3.zip"), "wb") as f_out:
                for file in files[:3]:
                    part_path = self.path.joinpath(file)
                    with open(part_path, "rb") as f_in:
                        f_out.write(f_in.read())
            
            # Unzip the concatenated ZIP file
            audio_folder = self.path.joinpath("audio")
            audio_folder.mkdir(parents=True)
            with zipfile.ZipFile(self.path.joinpath("mp3.zip"), "r") as z:
                z.extractall(audio_folder)
                
            # Remove ZIP files
            self.path.joinpath("mp3.zip").unlink()
            self.path.joinpath("mp3.zip.001").unlink()
            self.path.joinpath("mp3.zip.002").unlink()
            self.path.joinpath("mp3.zip.003").unlink()
            print(f"Files unzipped to '{audio_folder}'.")

        else:
            print(f"Dataset already downloaded. Remove '{self.path}' to download again.")

    def _create_targets(self) -> None:
        ann = pd.read_csv(self.path.joinpath("annotations_final.csv"), sep="\t")
        ann = ann.dropna(subset="mp3_path").reset_index(drop=True)

        # Get path to audio files
        self.audio_files = list(ann["mp3_path"].values)

        # Select the n most tags as targets
        tag_counts = ann.drop(["clip_id", "mp3_path"], axis=1).sum().sort_values(ascending=False)
        self.class_names = tag_counts[:self.num_classes].index.to_list()
        self.targets = torch.tensor(ann[self.class_names].values, dtype=torch.int8)

    def random_split(self, fractions: tuple = (0.75, 0.10, 0.15), 
                     random_state: None | int = None) -> tuple:
        
        clip_info = pd.read_csv(self.path.joinpath("clip_info_final.csv"), sep="\t")
        clip_info = clip_info.dropna(subset="mp3_path").reset_index(drop=True)
        grouped_clips = clip_info.groupby(
            ["track_number", "title", "artist", "album", "url", "original_url"]
        )
        grouped_clips = list(grouped_clips.groups.values())
        random.seed(random_state)
        random.shuffle(grouped_clips)

        num_tracks = len(grouped_clips)
        train_split = int(num_tracks * fractions[0])
        val_split = int(num_tracks * fractions[1])

        train_idx = grouped_clips[:train_split]
        val_idx = grouped_clips[train_split : (train_split + val_split)]
        test_idx = grouped_clips[(train_split + val_split) :]    

        train_idx = [int(i) for idx in train_idx for i in idx]
        val_idx = [int(i) for idx in val_idx for i in idx]        
        test_idx = [int(i) for idx in test_idx for i in idx]
        
        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)