import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy

from sound_stamp.networks import MusicTaggerFCN


class MusicTagger:
    
    def __init__(self, class_names: list[str]) -> None:
        self.model = MusicTaggerFCN(num_classes=len(class_names))
        self.class_names = class_names
                
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.model.to(self.device)
    
    def train(self, loader: DataLoader, learning_rate: float) -> float:
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        total_loss = 0.0
        
        self.model.train()
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(features)
            loss = binary_cross_entropy(output, targets)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()          
            
        return total_loss / len(loader)
    
    def evaluate(self, loader: DataLoader) -> float:
        total_loss = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(features)
                total_loss += binary_cross_entropy(output, targets).item()
            
        return total_loss / len(loader)
    
    def inference(self, features: torch.Tensor) -> torch.Tensor:       
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            output = self.model(features)
        
        return output
    
    def save(self, file_name: str, verbose: bool = True) -> None:
        torch.save(self.model.state_dict(), file_name)
        if verbose:
            print(f"Model parameters saved to '{file_name}'.")
        
    def load(self, file_name: str, verbose: bool = True) -> None:
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))
        if verbose:
            print(f"Model parameters loaded from '{file_name}'.")
