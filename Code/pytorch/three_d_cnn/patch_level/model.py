from torchvision.models import alexnet
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from modules import *
from copy import deepcopy
from classification_utils import *

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

###########################################################
### Model training from scratch
###########################################################

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
        #self.fc1 = nn.Linear(in_features=65536, out_features=2)
        ## for input size 51, 51
        self.fc1 = nn.Linear(in_features=128, out_features=2)
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
        print("For specific input size, to check the dimension to flatten the features for FC layer, dimension is: %s" % str(out.shape))
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
    Ref: `Automated classification of Alzheimer's disease and mild cognitive impairment using a single MRI and deep neural networks`
    """
    def __init__(self, n_classes=2, **kwargs):
        super(AllConvNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 50, 5, stride=1)
        self.bn1 = nn.BatchNorm3d(num_features=50)
        self.conv2 = nn.Conv3d(50, 100, 5, stride=2)
        self.bn2 = nn.BatchNorm3d(num_features=100)
        self.conv3 = nn.Conv3d(100, 200, 3, stride=1)
        self.bn3 = nn.BatchNorm3d(num_features=200)
        self.conv4 = nn.Conv3d(200, 300, 3, stride=2)
        self.bn4 = nn.BatchNorm3d(num_features=300)
        self.conv5 = nn.Conv3d(300, 400, 3, stride=1)
        self.bn5 = nn.BatchNorm3d(num_features=400)
        self.conv6 = nn.Conv3d(400, 500, 3, stride=2)
        self.bn6 = nn.BatchNorm3d(num_features=500)
        #self.conv7 = nn.Conv3d(500, 600, 3, stride=1)
        #self.conv8 = nn.Conv3d(600, 700, 3, stride=2)
        #self.conv9 = nn.Conv3d(700, 800, 3, stride=1)
        #self.conv10 = nn.Conv3d(800, 900, 3, stride=2)
        #self.conv11 = nn.Conv3d(900, 1000, 3, stride=1)
        #self.conv12 = nn.Conv3d(1000, 1100, 3, stride=2)
        self.class_conv = nn.Conv3d(500, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #x_drop = F.dropout(x, .2)
        conv1_out = self.bn1(F.leaky_relu(self.conv1(x)))
        conv2_out = self.bn2(F.leaky_relu(self.conv2(conv1_out)))
        conv3_out = self.bn3(F.leaky_relu(self.conv3(conv2_out)))
        #conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = self.bn4(F.leaky_relu(self.conv4(conv3_out)))
        conv5_out = self.bn5(F.leaky_relu(self.conv5(conv4_out)))
        conv6_out = self.bn6(F.leaky_relu(self.conv6(conv5_out)))
        #conv6_out_drop = F.dropout(conv6_out, .5)
        #conv7_out = F.relu(self.conv7(conv6_out_drop))
        #conv8_out = F.relu(self.conv8(conv7_out))
        #conv9_out = F.relu(self.conv9(conv8_out))
        #conv9_out_drop = F.dropout(conv9_out, .5)
        # #conv10_out = F.relu(self.conv10(conv9_out_drop))
        #conv11_out = F.relu(self.conv11(conv10_out))
        #conv12_out = F.relu(self.conv12(conv11_out))

        class_out = F.leaky_relu(self.class_conv(conv6_out))

        pool_out = F.adaptive_avg_pool3d(class_out, 1)
        #pool_out.squeeze_(-1)
        #pool_out.squeeze_(-1)
        #pool_out.squeeze_(-1)
        pool_out = pool_out.view(pool_out.size(0), -1)
        out = self.softmax(pool_out)

        return out

###########################################################
### Model for autoencoder pretraining
###########################################################

class Conv_5_FC_2(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_5_FC_2, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 5 * 6 * 5, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 4, 5, 4, 23]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

############################################
### Autoencoder
############################################

# class SparseAutoencoder(nn.Module):
#     """
#     This is the implementation of SparseAutoencoder.
#     Ideally, we can train each layer of a CNN using this ae and transfer the learned parameters to the task-specific CNN
#     Note: Need to calculate the size of in_features and out_features of the nn.Linear layer to fit each layer of CNN.
#
#     Ref of the sparse ae: https://stats.stackexchange.com/questions/149478/what-is-the-intuition-behind-the-sparsity-parameter-in-sparse-autoencoders
#                     paper: `A fast learning algorithm for deep belief nets`
#
#     # How to choose the hyperparameter of numbers of hidden layer units: https://stats.stackexchange.com/questions/101237/sparse-autoencoder-hyperparameters
#     """
#
#     def __init__(self, input_size, **kwargs):
#         super(SparseAutoencoder, self).__init__()
#         self.input_size = input_size
#         self.encoder = nn.Linear(int(input_size * input_size * input_size), int(input_size * input_size * input_size * 1.5))
#         self.decoder = nn.Linear(int(input_size * input_size * input_size * 1.5), int(input_size * input_size * input_size))
#
#     def forward(self, x):
#         out = x.view(-1, self.input_size * self.input_size * self.input_size)
#         encoded = F.sigmoid(self.encoder(out))
#         decoded = F.sigmoid(self.decoder(encoded))
#         return decoded, encoded
#
# ############################################
# ### Stacked Convoultional Autoencoder
# ############################################
#
# class ConvAutoencoder(nn.Module):
#     r"""
#     Convolutional autoencoder layer for stacked autoencoders.
#     This module is automatically trained when in model.training is True.
#
#     Ref: `Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction`,
#          `Reducing the Dimensionality of Data with Neural Networks`,
#          `Extracting and Composing Robust Features with Denoising Autoencoders`
#
#     Args:
#         input_size: The number of features in the input
#         output_size: The number of features to output
#         stride: Stride of the convolutional layers.
#     """
#
#     def __init__(self, input_size, output_size):
#         super(ConvAutoencoder, self).__init__()
#
#         self.forward_pass = nn.Sequential(
#             nn.Conv3d(input_size, output_size, kernel_size=2, stride=2, padding=0),
#             nn.ReLU(),
#             nn.MaxPool3d(2, stride=2, return_indices=True)
#         )
#         self.backward_pass = nn.Sequential(
#             nn.ConvTranspose3d(output_size, input_size, kernel_size=2, stride=2, padding=0),
#             nn.ReLU(),
#         )
#
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
#         self.max_pool_indices=None
#
#     def forward(self, x):
#         # Train each autoencoder with backpropogation if model is set to be train
#         x = x.detach()
#         # y, self.max_pool_indices = self.forward_pass(x)
#         # Add noise for the input.
#         x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
#         y, self.max_pool_indices = self.forward_pass(x_noisy)
#
#         if self.training:
#             ## check if the original shape of x is the exponential of 2, assuming that the 3 dimensions of patches are the same.
#             if (x.shape[-1]) & (x.shape[-1] - 1) == 0:
#                 x_reconstruct = F.max_unpool3d(y, self.max_pool_indices, 2, 2)
#             else:
#                 x_reconstruct = F.max_unpool3d(y, self.max_pool_indices, 2, 2, output_size=[2 * y.shape[-1] + 1, 2 * y.shape[-1] + 1, 2 * y.shape[-1] + 1])
#             x_reconstruct = self.backward_pass(x_reconstruct)
#             loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#         return y.detach()
#
#     def reconstruct(self, x):
#
#         ## check if the original shape of x is the exponential of 2, assuming that the 3 dimensions of patches are the same.
#         if (x.shape[-1]) & (x.shape[-1] - 1) == 0:
#             x_reconstruct = F.max_unpool3d(x, self.max_pool_indices, 2, 2)
#         else:
#             x_reconstruct = F.max_unpool3d(x, self.max_pool_indices, 2, 2,
#                                            output_size=[2 * x.shape[-1] + 1, 2 * x.shape[-1] + 1, 2 * x.shape[-1] + 1])
#         reconstruct = self.backward_pass(x_reconstruct)
#
#         return reconstruct
#
#
# class StackedConvDenAutoencoder(nn.Module):
#     r"""
#     A stacked autoencoder made from the convolutional denoising autoencoders above.
#     Each autoencoder is trained independently and at the same time.
#
#     Note: as for each AE, we use one conv layer and one maxpooling, so that the encoder of each AE will outputs 1/4
#     dimension patches of the original one.
#     """
#
#     def __init__(self):
#         super(StackedConvDenAutoencoder, self).__init__()
#
#         self.ae1 = ConvAutoencoder(1, 128)
#         self.ae2 = ConvAutoencoder(128, 256)
#         self.ae3 = ConvAutoencoder(256, 512)
#
#     def forward(self, x):
#         a1 = self.ae1(x)
#         a2 = self.ae2(a1)
#         a3 = self.ae3(a2)
#
#         if self.training:
#             return a3
#
#         else:
#             return a3, self.reconstruct(a3)
#
#     def reconstruct(self, x):
#         a2_reconstruct = self.ae3.reconstruct(x)
#         a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
#         x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
#         return x_reconstruct

class AutoEncoder(nn.Module):
    """
    This is a class to reconstruct the corresponding AE based on the given CNN. Basically, the CNN will become the encoder
    of the AE and the decoder part will be reconstructed.

    TODO: this class strictly need you form your CNN like this: ConvNet + BN + ReLu + MaxPool
    """

    def __init__(self, model=None):
        super(AutoEncoder, self).__init__()

        self.level = 0 ## the number of layer based on the convnet

        if model is not None:
            self.encoder = deepcopy(model.features)
            self.decoder = self.construct_decoder_layers(model) ## construct the decoder part

            for i, layer in enumerate(self.encoder):
                if isinstance(layer, PadMaxPool3d):
                    self.encoder[i].set_new_return()
                elif isinstance(layer, nn.MaxPool3d):
                    self.encoder[i].return_indices = True
        else:
            self.encoder = nn.Sequential() # This is used to reconstruct each AE for layer-wise training
            self.decoder = nn.Sequential()

    def __len__(self):
        return len(self.encoder)

    def forward(self, x):

        indices_list = []
        pad_list = []
        # If your version of Pytorch <= 0.4.0 you can execute this method on a GPU
        for layer in self.encoder:
            if isinstance(layer, PadMaxPool3d):
                x, indices, pad = layer(x)
                indices_list.append(indices)
                pad_list.append(pad)
            elif isinstance(layer, nn.MaxPool3d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)

        for layer in self.decoder:
            if isinstance(layer, CropMaxUnpool3d):
                x = layer(x, indices_list.pop(), pad_list.pop())
            elif isinstance(layer, nn.MaxUnpool3d):
                x = layer(x, indices_list.pop())
            else:
                x = layer(x)

        return x

    def construct_decoder_layers(self, model):
        """
        Reconstruct the decoder part of the AD based on the given CNN
        :param model: The target CNN architecture
        :return: the decoder
        """
        decoder_layers = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv3d):
                decoder_layers.append(nn.ConvTranspose3d(layer.out_channels, layer.in_channels, layer.kernel_size,
                                                     stride=layer.stride, padding=layer.padding))
                self.level += 1
            elif isinstance(layer, PadMaxPool3d):
                decoder_layers.append(CropMaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.MaxPool3d):
                decoder_layers.append(nn.MaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.Linear):
                decoder_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, Flatten):
                decoder_layers.append(Reshape(model.flattened_shape))
            elif isinstance(layer, nn.LeakyReLU):
                decoder_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
            # elif i == len(self.encoder) - 1 and isinstance(layer, nn.BatchNorm3d):
            #     pass
            else:
                decoder_layers.append(deepcopy(layer))
        decoder_layers = replace_relu(decoder_layers) ## Reverse the ReLu and ConvNet layers, it seems give better results
        decoder_layers.reverse()
        return nn.Sequential(*decoder_layers)

