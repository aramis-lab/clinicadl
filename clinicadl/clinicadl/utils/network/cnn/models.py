import numpy as np
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock

from clinicadl.utils.network.cnn.resnet import ResNetDesigner, model_urls
from clinicadl.utils.network.network import CNN
from clinicadl.utils.network.network_utils import *  # TODO: remove EarlyStopping from network_utils


class Conv5_FC3(CNN):
    """
    Classifier for a binary classification task

    Image level architecture
    """

    def __init__(self, input_shape, use_cpu=False, n_classes=2, dropout=0.5):
        # fmt: off
        convolutions = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_shape).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, n_classes)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions, classifier=classifier, use_cpu=use_cpu
        )


class Conv4_FC3(CNN):
    """
    Classifier for a binary classification task

    Image level architecture
    """

    def __init__(self, input_shape, use_cpu=False, n_classes=2, dropout=0.5):
        # fmt: off
        convolutions = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_shape).unsqueeze(0)
        output_convolutions = convolutions(input_tensor)

        classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)), 50),
            nn.ReLU(),

            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes)
        )
        # fmt: on
        super().__init__(
            convolutions=convolutions, classifier=classifier, use_cpu=use_cpu
        )


class resnet18(CNN):
    def __init__(self, use_cpu=False, n_classes=2, dropout=0.5):
        model = ResNetDesigner(BasicBlock, [2, 2, 2, 2])
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))

        convolutions = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )

        # add a fc layer on top of the transfer_learning model and a softmax classifier
        classifier = nn.Sequential(nn.Flatten(), model.fc)
        classifier.add_module("drop_out", nn.Dropout(p=dropout))
        classifier.add_module("fc_out", nn.Linear(1000, n_classes))

        super().__init__(
            convolutions=convolutions, classifier=classifier, use_cpu=use_cpu
        )
