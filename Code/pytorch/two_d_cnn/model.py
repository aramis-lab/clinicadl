import torch.nn as nn
from torchvision.models import alexnet

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

class LenetAdopted2D(nn.Module):
    """
    Pytorch implementation of customized Lenet-5.
    The original Lenet-5 architecture was described in the original paper `"Gradient-Based Learning Applied to Document Recgonition`.
        The original architecture includes: input_layer + conv1 + maxpool1 + conv2 + maxpool2 + fc1 + fc2 + output, activation function function used is relu.

    In our implementation, we adopted batch normalization layer and dropout techniques, we chose to use leaky_relu for the activation function.

    To note:
        Here, we train it from scratch and use the original signal of MRI slices, which shape is (H * W * 1), thus the in_channels is 1

    """

    def __init__(self, mri_plane, num_classes=2):
        super(LenetAdopted2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if mri_plane == 0:
            self.classifier = nn.Sequential(
                nn.Linear(61 * 54 * 43, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                nn.LogSoftmax()
            )
        elif mri_plane == 1:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 43, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                nn.LogSoftmax()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 51, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                nn.LogSoftmax()
            )


    def forward(self, x):
        """
        This is the function to pass the image tensor into the model
        :param x:
        :return:
        """
        x = self.features(x)
        x = x.view(x.size(0), -1) ## reshape the tensor so that it can be connected with self.classifier()
        x = self.classifier(x)
        return x


class AlexNetOneChannel(nn.Module):

    def __init__(self, mri_plane, num_classes=1000):
        super(AlexNetOneChannel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        if mri_plane == 0:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 5 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        elif mri_plane == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 4, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 4 * 5, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet2D(mri_plane=0, pretrained=False, **kwargs):
    """Implementation of AlexNet model architecture based on this paper: `"One weird trick..." <https://arxiv.org/abs/1404.5997>`.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """

    if pretrained == True:
        model = alexnet(pretrained)
        for p in model.features.parameters():
            p.requires_grad = False

        # fine-tune the last convolution layer
        for p in model.features[10].parameters():
            p.requires_grad = True
        # fine-tune the last second convolution layer
        for p in model.features[8].parameters():
            p.requires_grad = True

        ## add a fc layer on top of the pretrained model and a sigmoid classifier
        model.classifier.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.classifier.add_module('sigmoid', nn.LogSoftmax())

    else:
        model = AlexNetOneChannel(mri_plane, num_classes=2)

    return model
