import torch.nn as nn
from torchvision.models import alexnet
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

############################################
### LeNet
############################################

class lenet2D(nn.Module):
    """
    Pytorch implementation of customized Lenet-5.
    The original Lenet-5 architecture was described in the original paper `"Gradient-Based Learning Applied to Document Recgonition`.
        The original architecture includes: input_layer + conv1 + maxpool1 + conv2 + maxpool2 + fc1 + fc2 + output, activation function function used is relu.

    In our implementation, we adopted batch normalization layer and dropout techniques, we chose to use leaky_relu for the activation function.

    To note:
        Here, we train it from scratch and use the original signal of MRI slices, which shape is (H * W * 1), thus the in_channels is 1

    """

    def __init__(self, mri_plane, num_classes=2):
        super(lenet2D, self).__init__()
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
                nn.Linear(64 * 51 * 43, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                nn.Softmax(dim=1)
            )
        elif mri_plane == 1:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 43, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 51, 512),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(512, num_classes),
                # the Softmax has been encompassed into the loss function in Pytorch implementation, if you just wanna the label, it does not change anything
                # for the classification, because you will call argmax on the logits; otherwise, if you want to have a probability, you should always add a softmax
                # layer
                nn.Softmax(dim=1)
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


############################################
### AlexNet
############################################

class alexnetonechannel(nn.Module):
    """
    In the implementation of torchvision, the softmax layer was encompassed in the loss function 'CrossEntropyLoss' and
    'NLLLoss'
    """

    def __init__(self, mri_plane, num_classes=1000):
        super(alexnetonechannel, self).__init__()
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

        ## fine-tune the last convolution layer
        for p in model.features[10].parameters():
            p.requires_grad = True
        # fine-tune the last second convolution layer
        for p in model.features[8].parameters():
            p.requires_grad = True

        ## add a fc layer on top of the pretrained model and a sigmoid classifier
        model.classifier.add_module('dropout', nn.Dropout(p=0.8))
        model.classifier.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.classifier.add_module('sigmoid', nn.Softmax(dim=1))

    else:
        model = alexnetonechannel(mri_plane, num_classes=2)

    return model

############################################
### ResNet
############################################

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
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
        x = self.fc_out(x)
        x = self.sigmoid(x)

        return x

def resnet2D(resnet_type, pretrained=False, **kwargs):
    """
    Construct a RestNet model, the type of resnet models were list as variable: resnet_type.

    :param resnet_type: One of these models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param kwargs:
    :return:
    """
    if resnet_type == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the pretrained model and a sigmoid classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('sigmoid', nn.Softmax(dim=1))

    elif resnet_type == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the pretrained model and a sigmoid classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('sigmoid', nn.Softmax(dim=1))

    elif resnet_type == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the pretrained model and a sigmoid classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('sigmoid', nn.Softmax(dim=1))

    elif resnet_type == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the pretrained model and a sigmoid classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('sigmoid', nn.Softmax(dim=1))

    elif resnet_type == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the pretrained model and a sigmoid classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('sigmoid', nn.Softmax(dim=1))

    return model


############################################
### AllConvNet
############################################

import torch.nn as nn

class AllConvNet2D(nn.Module):
    """
    Pytorch implementation of `Striving for Simplicity: The All Convolutional Net` (https://arxiv.org/abs/1412.6806)
    """
    def __init__(self, input_size, n_classes=2, **kwargs):
        super(AllConvNet2D, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.class_conv = nn.Conv2d(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)

        return pool_out