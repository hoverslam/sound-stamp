import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from sound_stamp.datasets import MagnaSet
from sound_stamp.tagger import MusicTagger
from sound_stamp.utils import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training loop for music tagger.")
    parser.add_argument("-m", "--model", type=str, default="music_tagger", metavar="",
                        help="Name of the file to save the model. Defaults to 'music_tagger'.")
    args = parser.parse_args()
    model = args.model
    
    # Load configs
    configs = load_yaml(Path.cwd().joinpath("configs", "MusicTaggerFCN.yaml"))
    num_epochs = configs["hyperparameters"]["num_epochs"]
    batch_size = configs["hyperparameters"]["batch_size"]
    learning_rate = configs["hyperparameters"]["learning_rate"]
    patience = configs["hyperparameters"]["patience"]
    refinement = configs["hyperparameters"]["refinement"]
    refinement_lr_factor = configs["hyperparameters"]["refinement_lr_factor"]
    
    # Load data set
    dataset = MagnaSet(configs)
    train, val, _ = dataset.random_split(random_state=42)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

    # Initialize model
    tagger = MusicTagger(configs)

    # Training loop
    print(f"Training on {tagger.device} ...")
    best_val_loss = float("inf")
    cur_patience = patience
    cur_refinement = refinement
    cur_learning_rate = learning_rate
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = tagger.train(train_loader, cur_learning_rate)
        val_loss = tagger.evaluate(val_loader)
        print(f"Average loss: train={train_loss:.4f}, val={val_loss:.4f}")
        
        # Early stopping with refinement
        if val_loss < best_val_loss:
            tagger.save("models/checkpoint.pt", verbose=False)
            best_val_loss = val_loss
            cur_patience = patience        
        else:
            if cur_patience == 0:
                if cur_refinement > 0:
                    tagger.load("models/checkpoint.pt", verbose=False)
                    cur_learning_rate = cur_learning_rate * refinement_lr_factor                
                    cur_patience = patience
                    cur_refinement -= 1
                    print(f"Refinement with new learning rate {cur_learning_rate:.6f}.")
                else:
                    print("Stopped early!")
                    break
            else:
                cur_patience -= 1    

    # Save the best model with a new name
    tagger.load("models/checkpoint.pt", verbose=False)
    tagger.save(f"models/{model}.pt")