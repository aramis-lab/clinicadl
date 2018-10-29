import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init

from torchvision import models
from torch.nn.parameter import Parameter

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNetQC(nn.Module):

    def __init__(self, version=1.0, num_classes=2, use_ref=False):
        super(SqueezeNetQC, self).__init__()
        self.use_ref = use_ref
        self.feat = 3
        
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(2 if use_ref else 1, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(2 if use_ref else 1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512*self.feat, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # split feats into batches, so each view is passed separately
        x = x.view(-1, 2 if self.use_ref else 1 ,224,224)
        x = self.features(x)
        # reshape input to take into account 3 views
        x = x.view(-1, 512*self.feat,13,13)
        
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

    def load_from_std(self, std_model):
        # import weights from the standard ResNet model
        # TODO: finish
        # first load all standard items
        own_state = self.state_dict()
        for name, param in std_model.state_dict().items():
            if name == 'features.0.weight':
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
            elif name == 'classifier.1.weight' or name == 'classifier.1.bias':
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
            


def squeezenet_qc(pretrained=False, **kwargs):
    """Constructs a SqueezeNet 1.1 model

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNetQC(version=1.1, **kwargs)
    if pretrained:
        # load basic Resnet model
        model_ft = models.squeezenet1_1(pretrained=True)
        model.load_from_std(model_ft)
    return model
