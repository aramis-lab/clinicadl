import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from torchvision import models
from torch.nn.parameter import Parameter

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class ResNetQC(nn.Module):

    def __init__(self, block, layers, num_classes=2, use_ref=False):
        self.inplanes = 64
        self.use_ref = use_ref
        self.feat = 3
        self.expansion = block.expansion
        super(ResNetQC, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(2 if self.use_ref else 1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for merging multiple features
        self.addon = nn.Sequential(
            nn.Conv2d(self.feat*512*block.expansion, 512*block.expansion, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512*block.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(512*block.expansion, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0,bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5,inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
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
        # split feats into batches
        x = x.view(-1, 2 if self.use_ref else 1 ,224,224)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # merge batches together
        x = x.view(-1, 512*self.feat,7,7)
        x = self.addon(x)
        x = x.view(x.size(0), -1)

        return x

    def load_from_std(self, std_model):
        # import weights from the standard ResNet model
        # TODO: finish
        # first load all standard items
        own_state = self.state_dict()
        for name, param in std_model.state_dict().items():
            if name == 'conv1.weight':
                if isinstance(param, Parameter):
                    param = param.data
                # convert to mono weight
                # collaps parameters along second dimension, emulating grayscale feature 
                mono_param=param.sum( 1, keepdim=True )
                if self.use_ref:
                    own_state[name].copy_( torch.cat((mono_param,mono_param),1) )
                else:
                    own_state[name].copy_( mono_param )
                pass
            elif name == 'fc.weight' or name == 'fc.bias' or name == 'conv2.weight' or name == 'conv2.bias':
                # don't use at all
                pass
            elif name in own_state:
                if isinstance(param, Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            




def resnet_qc_18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQC(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet18(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQC(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.resnet34(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQC(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_ft = models.resnet50(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQC(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_ft = models.resnet101(pretrained=True)
        model.load_from_std(model_ft)
    return model


def resnet_qc_152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetQC(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_ft = models.resnet152(pretrained=True)
        model.load_from_std(model_ft)
    return model
