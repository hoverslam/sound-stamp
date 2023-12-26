import argparse

from torch.utils.data import DataLoader

from sound_stamp.datasets import MagnaSet
from sound_stamp.tagger import MusicTagger


# Settings
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 10
PATIENCE = 3
REFINEMENT = 2
REFINEMENT_LR_FACTOR = 0.1
LEARNING_RATE = 3e-4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training loop for music tagger.")
    parser.add_argument("--file_name", "-f", type=str, default="music_tagger.pt",
                        help="Name of the file to save the model. Defaults to 'music_tagger.pt'.")
    args = parser.parse_args()
    file_name = args.file_name
    
    # Load data set
    dataset = MagnaSet()
    train, val, _ = dataset.random_split(random_state=SEED)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    tagger = MusicTagger(dataset.class_names)

    # Training loop
    print("Training ...")

    best_val_loss = float("inf")
    cur_patience = PATIENCE
    cur_refinement = REFINEMENT
    cur_learning_rate = LEARNING_RATE

    for epoch in range(NUM_EPOCHS):
        train_loss = tagger.train(train_loader, cur_learning_rate)
        val_loss = tagger.evaluate(val_loader)
        print(f"{epoch+1}/{NUM_EPOCHS}: {train_loss=:.4f}, {val_loss=:.4f}")
        
        # Early stopping with refinement
        if val_loss < best_val_loss:
            tagger.save("models/checkpoint.pt", verbose=False)
            best_val_loss = val_loss
            cur_patience = PATIENCE        
        else:
            if cur_patience == 0:
                if cur_refinement > 0:
                    tagger.load("models/checkpoint.pt", verbose=False)
                    cur_learning_rate = cur_learning_rate * REFINEMENT_LR_FACTOR                
                    cur_patience = PATIENCE
                    cur_refinement -= 1
                    print(f"Refinement with new learning rate {cur_learning_rate}.")
                else:
                    print("Stopped early!")
                    break
            else:
                cur_patience -= 1    

    # Save the best model with a new name
    tagger.load("models/checkpoint.pt", verbose=False)
    tagger.save(f"models/{file_name}")