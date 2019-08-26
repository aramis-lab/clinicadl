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
### Model for autoencoder pretraining
###########################################################

class Conv_4_FC_3(nn.Module):
    """
       This network is the implementation of this paper:
       'Multi-modality cascaded convolutional neural networks for Alzheimer's Disease diagnosis'
       """

    def __init__(self, dropout=0, n_classes=2):
        super(Conv_4_FC_3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 15, 3),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3),
            nn.BatchNorm3d(25),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(50, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(50 * 2 * 2 * 2, 50),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes),
            nn.Softmax(dim=1)
        )

        self.flattened_shape = [-1, 50, 2, 2, 2]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_3_FC_2(nn.Module):
    """
       """

    def __init__(self, dropout=0, n_classes=2):
        super(Conv_3_FC_2, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 15, 3),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(15, 25, 3),
            nn.BatchNorm3d(25),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(25, 50, 3),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            # nn.Linear(50 * 5 * 5 * 5, 50),
            nn.Linear(50 * 3 * 3 * 3, 50),
            nn.ReLU(),

            nn.Dropout(p=dropout),
            nn.Linear(50, 40),
            nn.ReLU(),

            nn.Linear(40, n_classes),
            nn.Softmax(dim=1)
        )

        # self.flattened_shape = [-1, 50, 5, 5, 5]
        ## for hippocampus
        self.flattened_shape = [-1, 50, 3, 3, 3]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

class Conv_7_FC_2(nn.Module):
    """
       Classification model based on the input patch.
       """

    def __init__(self, dropout=0.5, n_classes=2):
        super(Conv_7_FC_2, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.Conv3d(16, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.Conv3d(64, 64, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 1 * 1 * 1, 100),
            nn.ReLU(),

            # nn.Dropout(p=dropout),
            # nn.Linear(1000, 100),
            # nn.ReLU(),

            nn.Linear(100, n_classes),
            nn.Softmax(dim=1)
        )

        self.flattened_shape = [-1, 64, 1, 1, 1]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

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
        decoder_layers = revese_relu_conv(decoder_layers) ## Reverse the ReLu and ConvNet layers, it seems give better results
        decoder_layers.reverse()
        return nn.Sequential(*decoder_layers)
