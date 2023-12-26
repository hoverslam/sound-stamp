import torch
from torch import nn


class MusicTaggerFCN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
             
        self.conv = nn.Sequential(
            # 1st block
            nn.Conv2d(1, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 4), padding=1),
            nn.ELU(),
            # 2nd block
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 4), padding=1),
            nn.ELU(),
            # 3rd block
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 4), padding=1),
            nn.ELU(),
            # 4th block
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((3, 5), padding=1),
            nn.ELU(),
            # 5th block
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((4, 4), padding=1),
            nn.ELU()
        )

        self.linear = nn.Sequential(
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)    
        x = self.conv(x)
        x = x.view(-1, 64)      
        x = self.linear(x)

        return x
  