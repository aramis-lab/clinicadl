# coding: utf8

import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock
from torch import nn
import math

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}


def resnet18(**kwargs):
    """
    Construct a the ResNet-18 model with added dropout, FC and softmax layers.

    :param kwargs:
    :return:
    """
    model = ResNetDesigner(BasicBlock, [2, 2, 2, 2], **kwargs)
    try:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    except Exception as err:
        print("Error is:", err)
        # raise ConnectionError('The URL %s may not be functional anymore. Check if it still exists or '
        #                      'if it has been moved.' % model_urls['resnet18'])
    for p in model.parameters():
        p.requires_grad = False

    # fine-tune the 4-th residual block
    for p in model.layer4.parameters():
        p.requires_grad = True

    # fine-tune the last FC layer
    for p in model.fc.parameters():
        p.requires_grad = True

    # add a fc layer on top of the transfer_learning model and a softmax classifier
    model.add_module('drop_out', nn.Dropout(p=kwargs["dropout"]))
    model.add_module('fc_out', nn.Linear(1000, 2))

    return model


class ResNetDesigner(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNetDesigner, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Added top FC layer
        x = self.drop_out(x)
        x = self.fc_out(x)

        return x
