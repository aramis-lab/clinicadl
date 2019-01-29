import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import alexnet, vgg16
from models.incpetion_tl import *
from models.resnet_tl import *
from models.densenet_tl import *

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

############################################
### ImageNet pretrained models
############################################

############################################
### AlexNet
############################################

def AlexNet(transfer_learning=True, **kwargs):
    """Implementation of AlexNet model architecture based on this paper: `"One weird trick..." <https://arxiv.org/abs/1404.5997>`.

    Args:
        transfer_learning (bool): If True, returns a model transfer_learning on ImageNet
    """

    if transfer_learning == True:
        model = alexnet(transfer_learning)
        for p in model.features.parameters():
           p.requires_grad = False

        # fine tune the last tow conv layers
        for p in model.features[10].parameters():
            p.requires_grad = True
        for p in model.features[8].parameters():
            p.requires_grad = True

        ## fine-tune the classifer part of alexnet: 2 FC layer
        for p in model.classifier.parameters():
            p.requires_grad = True

        ## add a fc layer on top of the transfer_learning model and a softmax classifier
        ## alexnet has self.classifer in forward function, we do not have to rewrite the forward part for new adding layers.
        model.classifier.add_module('drop_out', nn.Dropout(p=0.8))
        model.classifier.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.classifier.add_module('softmax', nn.Softmax(dim=1))

    else:
        raise Exception("AlexNet is impossible to train from scracth in our application, you can implement it if you want!")

    return model

############################################
### Vgg net
############################################
def Vgg16(transfer_learning=True, **kwargs):
    """Transfer learning for VggNet.

    Args:
        transfer_learning (bool): If True, returns a model transfer_learning on ImageNet
    """

    if transfer_learning == True:
        model = vgg16(transfer_learning)
        for p in model.features.parameters():
            p.requires_grad = False

        ## fine-tune the last convnet features.28
        for p in model.features[28].parameters():
            p.requires_grad = True

        ## fine-tune the self.classifer containing 3 FC layers
        for p in model.classifier.parameters():
            p.requires_grad = True

        ## add a fc layer on top of the transfer_learning model and a softmax classifier
        ## Vgg16 has self.classifer in forward function, we do not have to rewrite the forward part for new adding layers.
        model.classifier.add_module('drop_out', nn.Dropout(p=0.8))
        model.classifier.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.classifier.add_module('softmax', nn.Softmax(dim=1))

    else:
        raise Exception("VggNet is impossible to train from scracth in our application, you can implement it if you want!")

    return model


############################################
### inception_v3
############################################
def InceptionV3(transfer_learning=True, **kwargs):
    """Transfer learning for Inception_v3.

    Args:
        transfer_learning (bool): If True, returns a model transfer_learning on ImageNet
    """

    if transfer_learning == True:
        model = inception_v3(transfer_learning)
        for p in model.parameters():
            p.requires_grad = False

        ## fine tune Mixed_7c block
        for p in model.Mixed_7c.parameters():
            p.requires_grad = True

        ## fine-tune the last fc layer
        for p in model.fc.parameters():
            p.requires_grad = True

        ## add a fc layer on top of the transfer_learning model and a softmax classifier
        model.add_module('drop_out', nn.Dropout(p=0.8))
        model.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.add_module('softmax', nn.Softmax(dim=1))

    else:
        raise Exception("InceptionV3 is impossible to train from scracth in our application, you can implement it if you want!")

    return model


############################################
### densenet161
############################################
def DenseNet161(transfer_learning=True, **kwargs):
    """Transfer learning for DenseNet161.

    Args:
        transfer_learning (bool): If True, returns a model transfer_learning on ImageNet
    """

    if transfer_learning == True:
        model = densenet161(transfer_learning)
        for p in model.features.parameters():
            p.requires_grad = False

        ### fine-tune the last dense block & last fc layer
        for p in model.features.denseblock4.denselayer24.parameters():
                p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        ## add a fc layer on top of the transfer_learning model and a softmax classifier
        ## DenseNet161 has self.classifer in forward function, we do not have to rewrite the forward part for new adding layers.
        model.add_module('drop_out', nn.Dropout(p=0.8))
        model.add_module('fc_out', nn.Linear(1000, 2)) ## For linear layer, Pytorch used similar H initialization for the weight.
        model.add_module('softmax', nn.Softmax(dim=1))

    else:
        raise Exception("DenseNet161 is impossible to train from scracth in our application, you can implement it if you want!")

    return model


############################################
### ResNets
############################################

def ResNet(resnet_type='resnet18', transfer_learning=True, **kwargs):
    """
    Construct a RestNet model, the type of resnet models were list as variable: resnet_type.

    :param resnet_type: One of these models: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    :param transfer_learning: If True, returns a model pre-trained on ImageNet
    :param kwargs:
    :return:
    """
    if resnet_type == 'resnet152':
        model = ResNets(Bottleneck, [3, 8, 36, 3], **kwargs)
        if transfer_learning:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the transfer_learning model and a softmax classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            raise Exception(
                "resnet152 is impossible to train from scracth in our application, you can implement it if you want!")

    elif resnet_type == 'resnet101':
        model = ResNets(Bottleneck, [3, 4, 23, 3], **kwargs)
        if transfer_learning:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the transfer_learning model and a softmax classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            raise Exception(
                "resnet101 is impossible to train from scracth in our application, you can implement it if you want!")

    elif resnet_type == 'resnet50':
        model = ResNets(Bottleneck, [3, 4, 6, 3], **kwargs)
        if transfer_learning:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the transfer_learning model and a softmax classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            raise Exception(
                "resnet50 is impossible to train from scracth in our application, you can implement it if you want!")

    elif resnet_type == 'resnet34':
        model = ResNets(BasicBlock, [3, 4, 6, 3], **kwargs)
        if transfer_learning:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the transfer_learning model and a softmax classifier
            model.add_module('fc_out', nn.Linear(1000, 2))  ## For linear layer, Pytorch used similar H initialization for the weight.
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            raise Exception(
                "resnet34 is impossible to train from scracth in our application, you can implement it if you want!")

    elif resnet_type == 'resnet18':
        model = ResNets(BasicBlock, [2, 2, 2, 2], **kwargs)
        if transfer_learning:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
            for p in model.parameters():
                p.requires_grad = False

            ## fine-tune the 4-th res blocak
            for p in model.layer4.parameters():
                p.requires_grad = True

            ## fine-tune the last FC layer
            for p in model.fc.parameters():
                p.requires_grad = True

            ## add a fc layer on top of the transfer_learning model and a softmax classifier
            model.add_module('drop_out', nn.Dropout(p=0.8))
            model.add_module('fc_out', nn.Linear(1000, 2))
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            raise Exception(
                "resnet18 is impossible to train from scracth in our application, you can implement it if you want!")

    return model

############################################
### Train from scratch models
############################################

############################################
### AllConvNet
############################################

import torch.nn as nn

class AllConvNet(nn.Module):
    """
    Pytorch implementation of `Striving for Simplicity: The All Convolutional Net` (https://arxiv.org/abs/1412.6806)
    """
    def __init__(self, input_size, n_classes=2, **kwargs):
        super(AllConvNet, self).__init__()
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

############################################
### LeNet
############################################

class LeNet(nn.Module):
    """
    Pytorch implementation of customized Lenet-5.
    The original Lenet-5 architecture was described in the original paper `"Gradient-Based Learning Applied to Document Recgonition`.
        The original architecture includes: input_layer + conv1 + maxpool1 + conv2 + maxpool2 + fc1 + fc2 + output, activation function function used is relu.

    In our implementation, we adopted batch normalization layer and dropout techniques, we chose to use leaky_relu for the activation function.

    To note:
        Here, we train it from scratch and use the original signal of MRI slices, which shape is (H * W * 1), thus the in_channels is 1

    """

    def __init__(self, mri_plane, num_classes=2):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.Dropout(0.8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Dropout(0.8),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if mri_plane == 0:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 51 * 43, 256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(0.8),
                nn.Linear(256, num_classes),
                nn.Softmax(dim=1)
            )
        elif mri_plane == 1:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 43, 256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, num_classes),
                nn.Softmax(dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(64 * 41 * 51, 256),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, num_classes),
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

class AlexNetonechannel(nn.Module):
    """
    In the implementation of torchvision, the softmax layer was encompassed in the loss function 'CrossEntropyLoss' and
    'NLLLoss'
    """

    def __init__(self, mri_plane, num_classes=2):
        super(AlexNetonechannel, self).__init__()
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
