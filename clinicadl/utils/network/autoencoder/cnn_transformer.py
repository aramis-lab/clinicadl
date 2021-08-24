from copy import deepcopy

from torch import nn

from clinicadl.utils.network.network_utils import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
    Reshape,
)


class CNN_Transformer(nn.Module):
    def __init__(self, model=None):
        """
        Construct an autoencoder from a given CNN. The encoder part corresponds to the convolutional part of the CNN.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        """
        from copy import deepcopy

        super(CNN_Transformer, self).__init__()

        self.level = 0

        if model is not None:
            self.encoder = deepcopy(model.convolutions)
            self.decoder = self.construct_inv_layers(model)

            for i, layer in enumerate(self.encoder):
                if isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d):
                    self.encoder[i].set_new_return()
                elif isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d):
                    self.encoder[i].return_indices = True
        else:
            self.encoder = nn.Sequential()
            self.decoder = nn.Sequential()

    def __len__(self):
        return len(self.encoder)

    def construct_inv_layers(self, model):
        """
        Implements the decoder part from the CNN. The decoder part is the symmetrical list of the encoder
        in which some layers are replaced by their transpose counterpart.
        ConvTranspose and ReLU layers are inverted in the end.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        :return: (Module) decoder part of the Autoencoder
        """
        inv_layers = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv3d):
                inv_layers.append(
                    nn.ConvTranspose3d(
                        layer.out_channels,
                        layer.in_channels,
                        layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
                )
                self.level += 1
            elif isinstance(layer, nn.Conv2d):
                inv_layers.append(
                    nn.ConvTranspose2d(
                        layer.out_channels,
                        layer.in_channels,
                        layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                    )
                )
                self.level += 1
            elif isinstance(layer, PadMaxPool3d):
                inv_layers.append(
                    CropMaxUnpool3d(layer.kernel_size, stride=layer.stride)
                )
            elif isinstance(layer, PadMaxPool2d):
                inv_layers.append(
                    CropMaxUnpool2d(layer.kernel_size, stride=layer.stride)
                )
            elif isinstance(layer, nn.Linear):
                inv_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, nn.Flatten):
                inv_layers.append(Reshape(model.flattened_shape))
            elif isinstance(layer, nn.LeakyReLU):
                inv_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
            else:
                inv_layers.append(deepcopy(layer))
        inv_layers = self.replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)

    @staticmethod
    def replace_relu(inv_layers):
        """
        Invert convolutional and ReLU layers (give empirical better results)

        :param inv_layers: (list) list of the layers of decoder part of the Auto-Encoder
        :return: (list) the layers with the inversion
        """
        idx_relu, idx_conv = -1, -1
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.ConvTranspose3d):
                idx_conv = idx
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                idx_relu = idx

            if idx_conv != -1 and idx_relu != -1:
                inv_layers[idx_relu], inv_layers[idx_conv] = (
                    inv_layers[idx_conv],
                    inv_layers[idx_relu],
                )
                idx_conv, idx_relu = -1, -1

        # Check if number of features of batch normalization layers is still correct
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.BatchNorm3d):
                conv = inv_layers[idx + 1]
                inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)

        return inv_layers
