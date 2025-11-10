import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, in_dim=512, num_modals=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_dim, num_modals),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.head(x)