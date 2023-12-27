import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAUROC

from sound_stamp.datasets import MagnaSet
from sound_stamp.tagger import MusicTagger
from sound_stamp.utils import load_yaml


# Load model configs
model_configs = load_yaml(Path.cwd().joinpath("configs", "models.yaml"))
batch_size = model_configs["MusicTaggerFCN"]["hyperparameters"]["batch_size"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a music tagger on the test set.")
    parser.add_argument("-m", "--model", type=str, default="music_tagger", metavar="",
                        help="File name of the trained model. Defaults to 'music_tagger'.")
    args = parser.parse_args()
    model = args.model
    
    # Load data set
    dataset = MagnaSet()
    _, _, test = dataset.random_split(random_state=42)
    test_loader = DataLoader(test, batch_size=batch_size)

    # Initialize model and load state dict
    tagger = MusicTagger(dataset.class_names)
    tagger.load(f"models/{model}.pt")

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