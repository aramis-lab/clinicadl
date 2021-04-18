# coding: utf8

"""
Script containing the models for the patch level experiments.
"""
from torch import nn
from .modules import PadMaxPool3d, Flatten

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


class Conv4_FC3(nn.Module):
    """
    Classifier for a binary classification task

    Patch level architecture used on Minimal preprocessing

    This network is the implementation of this paper:
    'Multi-modality cascaded convolutional neural networks for Alzheimer's Disease diagnosis'
    """

    def __init__(self, dropout=0, n_classes=2):
        super(Conv4_FC3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 15, 3),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3),
            nn.BatchNorm3d(25),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(50, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(50 * 2 * 2 * 2, 50),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes)
        )

        self.flattened_shape = [-1, 50, 2, 2, 2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
