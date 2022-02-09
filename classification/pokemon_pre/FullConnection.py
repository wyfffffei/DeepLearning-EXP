import torch.nn as nn


class FullConnection(nn.Module):
    def __init__(self):
        super(FullConnection, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(54, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
