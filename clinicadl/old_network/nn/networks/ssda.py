import numpy as np
import torch
import torch.nn as nn

from clinicadl.nn.layers import (
    GradientReversal,
    get_conv_layer,
    get_norm_layer,
    get_pool_layer,
)


class CNN_SSDA(nn.Module):
    """Base class for SSDA CNN."""

    def __init__(
        self,
        convolutions,
        fc_class_source,
        fc_class_target,
        fc_domain,
        alpha=1.0,
    ):
        super().__init__()
        self.convolutions = convolutions
        self.fc_class_source = fc_class_source
        self.fc_class_target = fc_class_target
        self.fc_domain = fc_domain
        self.grad_reverse = GradientReversal(alpha=alpha)

    def forward(self, x):
        x = self.convolutions(x)
        x_class_source = self.fc_class_source(x)
        x_class_target = self.fc_class_target(x)
        x_reverse = self.grad_reverse(x)
        x_domain = self.fc_domain(x_reverse)
        return x_class_source, x_class_target, x_domain


class Conv5_FC3_SSDA(CNN_SSDA):
    """
    Reduce the 2D or 3D input image to an array of size output_size.
    """

    def __init__(self, input_size, output_size=2, dropout=0.5, alpha=1.0):
        dim = len(input_size) - 1
        conv = get_conv_layer(dim)
        pool = get_pool_layer("PadMaxPool", dim=dim)
        norm = get_norm_layer("BatchNorm", dim=dim)

        convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),
            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),
            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),
            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),
            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        fc_class_source = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )
        fc_class_target = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )
        fc_domain = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),
            nn.Linear(1300, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
        )
        super().__init__(
            convolutions,
            fc_class_source,
            fc_class_target,
            fc_domain,
            alpha,
        )
