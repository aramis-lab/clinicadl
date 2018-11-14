import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


class AlexNet2D(nn.Module):
    """
    Pytorch implementation of AlexNet.

    To note: It contains 5 convolutional layers and 3 fully connected layers.
        Relu is applied after very convolutional and fully connected layer.
        Dropout is applied before the first and the second fully connected year.
        The image size in the following architecutre chart should be 227 * 227 instead of 224 * 224,
        as it is pointed out by Andrei Karpathy in his famous CS231n Course.
        More insterestingly, the input size is 224 * 224 with 2 padding in the pytorch torch vision.
        The output width and height should be 224 minus 11 plus 4 divide 4 plus 1 equel 55.25,
        The explanation here is pytorch Conv2d apply floor operator to the above result,
        and therefore the last one padding is ignored.

    """

    def __init__(self, num_classes=1000):
        super(AlexNet2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """
        This is the function to pass the image tensor into the model
        :param x:
        :return:
        """
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6) ## reshape the tensor so that it can be connected with self.classifier()
        x = self.classifier(x)
        return x


def alexnet2D(pretrained=False, **kwargs):
    """Implementation of AlexNet model architecture based on this paper: `"One weird trick..." <https://arxiv.org/abs/1404.5997>`.

    Args:
        pretrained (bool): If True, returns a model pretrained on ImageNet
    """

    pytorch_pretrained_alexnet_url = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}
    model = AlexNet2D(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(pytorch_pretrained_alexnet_url['alexnet']))
        for p in model.features.parameters():
            p.requires_grad = False ### download the pre-trained weighted from alexnet

        # fine-tune the last convolution layer
        for p in model.features[10].parameters():
            p.requires_grad = True
    ## add a fc layer on top of the pretrained model and a sigmoid classifier
    model.classifier.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
    model.classifier.add_module('sigmoid', nn.LogSoftmax())

    return model

## TODO implement Lenet-5 from scratch, which is not very deep.