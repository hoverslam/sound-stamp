import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC

from sound_stamp.datasets import MagnaSet
from sound_stamp.tagger import MusicTagger


# Settings
SEED = 42
BATCH_SIZE = 128

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a music tagger on the test set.")
    parser.add_argument("--file_name", "-f", type=str, default="music_tagger.pt",
                        help="File name of the trained model. Defaults to 'music_tagger.pt'.")
    args = parser.parse_args()
    file_name = args.file_name
    
    # Load data set
    dataset = MagnaSet()
    _, _, test = dataset.random_split(random_state=SEED)
    test_loader = DataLoader(test, batch_size=128)

    # Initialize model and load state dict
    tagger = MusicTagger(dataset.class_names)
    tagger.load(f"models/{file_name}")

    # Make predictions on test set
    print("Testing ...")
    predicted_tags = []
    true_tags = []

    for features, targets in test_loader:
        output = tagger.inference(features)
        predicted_tags.append(output)
        true_tags.append(targets)

    predicted_tags = torch.vstack(predicted_tags).cpu()
    true_tags = torch.vstack(true_tags).type(torch.int8).cpu()
    
    # Calculate Area under the ROC Curve (AUROC)
    auroc = MultilabelAUROC(num_labels=len(dataset.class_names), average="micro", thresholds=None)
    score = auroc(predicted_tags, true_tags).item()
    print(f"Area under the ROC Curve: {score:.4f}")