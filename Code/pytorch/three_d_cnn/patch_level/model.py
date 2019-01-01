from torchvision.models import alexnet
import torch.nn as nn
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
### VoxResNet
############################################
class Res_module(nn.Module):

    def __init__(self, features):
        super(Res_module, self).__init__()
        self.bn = nn.BatchNorm3d(num_features = features)
        self.conv = nn.Conv3d(in_channels = features, out_channels=features, kernel_size=3, stride=1, padding=1)
        
    def forward(self, out):
        out = F.relu(self.bn(out))
        out = F.relu(self.bn(self.conv(out)))
        out = self.conv(out)
        return out

class VoxResNet(nn.Module):
    """
    This is the implementation of VoxelResNet from this paper: `Deep voxelwise residual networks for volumetric brain segmentation`

    ## The orginal paper is for segmentation, if I should apply the 4 dconvolutional step ?
    """

    def __init__(self):
        super(VoxResNet, self).__init__()
        self.conv1_0 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_0 = nn.BatchNorm3d(num_features=32)
        self.bn1_1 = nn.BatchNorm3d(num_features=32)
        self.conv2_0 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.module1_0 = Res_module(features=64)
        self.module1_1 = Res_module(features=64)
        self.module1_2 = Res_module(features=64)
        self.module1_3 = Res_module(features=64)
        self.bn2_0 = nn.BatchNorm3d(num_features=64)
        self.bn2_1 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels =64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.module2_0 = Res_module(features=128)
        self.module2_1  = Res_module(features=128)
        self.pool = nn.MaxPool3d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(in_features=65536, out_features=2)
        # self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, out):
        out = F.relu(self.bn1_0(self.conv1_0(out)))
        out = F.relu(self.bn1_1(self.conv1_1(out)))

        out = self.conv2_0(out)
        out_s = self.module1_0(out)

        out_s = self.module1_1(out+out_s)

        out = F.relu(self.bn2_0(out+out_s))
        out = self.conv2_1(out)
        out_s = self.module1_2(out)

        out_s = self.module1_3(out+out_s)

        out = F.relu(self.bn2_1(out+out_s))
        out = self.conv3(out)
        out_s = self.module2_0(out)

        out_s = self.module2_1(out+out_s)

        out_= self.pool(out+out_s)
        out = out_.view(out_.size(0), -1)
        out = F.relu(self.fc1(out))
        # out = self.softmax(self.fc2(out))
        out = self.softmax(out)
        return out

############################################
### AllConvNet
############################################

import torch.nn as nn

class AllConvNet3D(nn.Module):
    """
    3D version of pytorch implementation of `Striving for Simplicity: The All Convolutional Net` (https://arxiv.org/abs/1412.6806)
    """
    def __init__(self, n_classes=2, **kwargs):
        super(AllConvNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 96, 3, padding=1)
        self.conv2 = nn.Conv3d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv3d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv3d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv3d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv3d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv3d(192, 192, 1)
        self.class_conv = nn.Conv3d(192, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

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

        pool_out = F.adaptive_avg_pool3d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)

        out = self.softmax(pool_out)

        return out

############################################
### Autoencoder
############################################

class SparseAutoencoder(nn.Module):
    """
    This is the implementation of SparseAutoencoder.
    Ideally, we can train each layer of a CNN using this ae and transfer the learned parameters to the task-specific CNN
    Note: Need to calculate the size of in_features and out_features of the nn.Linear layer to fit each layer of CNN.

    Ref of the sparse ae: https://stats.stackexchange.com/questions/149478/what-is-the-intuition-behind-the-sparsity-parameter-in-sparse-autoencoders
                    paper: `A fast learning algorithm for deep belief nets`

    # How to choose the hyperparameter of numbers of hidden layer units: https://stats.stackexchange.com/questions/101237/sparse-autoencoder-hyperparameters
    """

    def __init__(self, input_size, **kwargs):
        super(SparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.encoder = nn.Linear(int(input_size * input_size * input_size), int(input_size * input_size * input_size * 1.5))
        self.decoder = nn.Linear(int(input_size * input_size * input_size * 1.5), int(input_size * input_size * input_size))

    def forward(self, x):
        out = x.view(-1, self.input_size * self.input_size * self.input_size)
        encoded = F.sigmoid(self.encoder(out))
        decoded = F.sigmoid(self.decoder(encoded))
        return decoded, encoded

class autoencoder(nn.Module):
    """
    This is the implementation of convolutional autoencoder.

    Ref: `Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction`
    """
    def init(self):
        super(autoencoder, self).init()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid() ## value range [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
