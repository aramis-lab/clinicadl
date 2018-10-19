import torch.nn as nn
from modules import *

"""
All the architectures are built here
"""


class Hosseini(nn.Module):
    """
        Classifier for a multi-class classification task

        """

    def __init__(self, dropout=0.0, n_classes=2, negative_slope=0.01):
        super(Hosseini, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(8 * 14 * 17 * 14, 2000),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(2000, 500),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(500, n_classes)
        )

        self.flattened_shape = [-1, 8, 14, 17, 14]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
