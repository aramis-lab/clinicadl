from copy import deepcopy

import numpy as np
from torch import nn

from clinicadl.utils.network.models.modules import *


class RandomArchitecture(nn.Module):
    """
    Classifier for a multi-class classification task

    Initially named Initial_architecture
    """

    def __init__(
        self,
        convolutions,
        n_fcblocks,
        initial_shape,
        dropout=0.5,
        network_normalization="BatchNorm",
        n_classes=2,
    ):
        """
        Construct the Architecture randomly chosen for Random Search.

        Args:
            convolutions: (dict) description of the convolutional blocks.
            n_fcblocks: (int) number of FC blocks in the network.
            initial_shape: (list) gives the structure of the input of the network.
            dropout: (float) rate of the dropout.
            network_normalization: (str) type of normalization layer in the network.
            n_classes: (int) Number of output neurones of the network.
        """
        super(RandomArchitecture, self).__init__()
        self.dimension = len(initial_shape) - 1
        self.first_in_channels = initial_shape[0]
        self.layers_dict = self.return_layers_dict()
        self.features = nn.Sequential()
        self.network_normalization = network_normalization

        for key, item in convolutions.items():
            convolutional_block = self.define_convolutional_block(item)
            self.features.add_module(key, convolutional_block)

        self.classifier = nn.Sequential(Flatten(), nn.Dropout(p=dropout))

        fc, flattened_shape = self.fc_dict_design(
            n_fcblocks, convolutions, initial_shape, n_classes
        )
        for key, item in fc.items():
            n_fc = int(key[2::])
            if n_fc == len(fc) - 1:
                fc_block = self.define_fc_layer(item, last_block=True)
            else:
                fc_block = self.define_fc_layer(item, last_block=False)
            self.classifier.add_module(key, fc_block)

        self.flattened_shape = flattened_shape

    def __len__(self):
        fc_list = [
            ("classifier", "FC" + str(i)) for i in range(len(self.classifier) - 2)
        ]
        conv_list = [("features", "conv" + str(i)) for i in range(len(self.features))]
        return len(conv_list) + len(fc_list)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def define_convolutional_block(self, conv_dict):
        """
        Design a convolutional block from the dictionnary conv_dict.

        Args:
            conv_dict: (dict) A dictionnary with the specifications to build a convolutional block
            - n_conv (int) number of convolutional layers in the block
            - in_channels (int) number of input channels
            - out_channels (int) number of output channels (2 * in_channels or threshold = 512)
            - d_reduction (String) "MaxPooling" or "stride"
        Returns:
            (nn.Module) a list of modules in a nn.Sequential list
        """
        in_channels = (
            conv_dict["in_channels"]
            if conv_dict["in_channels"] is not None
            else self.first_in_channels
        )
        out_channels = conv_dict["out_channels"]

        conv_block = []
        for i in range(conv_dict["n_conv"] - 1):
            conv_block.append(
                self.layers_dict["Conv"](
                    in_channels, in_channels, 3, stride=1, padding=1
                )
            )
            conv_block = self.append_normalization_layer(conv_block, in_channels)
            conv_block.append(nn.LeakyReLU())
        if conv_dict["d_reduction"] == "MaxPooling":
            conv_block.append(
                self.layers_dict["Conv"](
                    in_channels, out_channels, 3, stride=1, padding=1
                )
            )
            conv_block = self.append_normalization_layer(conv_block, out_channels)
            conv_block.append(nn.LeakyReLU())
            conv_block.append(self.layers_dict["Pool"](2, 2))
        elif conv_dict["d_reduction"] == "stride":
            conv_block.append(
                self.layers_dict["Conv"](
                    in_channels, out_channels, 3, stride=2, padding=1
                )
            )
            conv_block = self.append_normalization_layer(conv_block, out_channels)
            conv_block.append(nn.LeakyReLU())
        else:
            raise ValueError(
                "Dimension reduction %s is not supported. Please only include"
                "'MaxPooling' or 'stride' in your sampling options."
                % conv_dict["d_reduction"]
            )

        return nn.Sequential(*conv_block)

    def append_normalization_layer(self, conv_block, num_features):
        """
        Appends or not a normalization layer to a convolutional block depending on network attributes.

        Args:
            conv_block: (list) list of the modules of the convolutional block
            num_features: (int) number of features to normalize
        Returns:
            (list) the updated convolutional block
        """

        if self.network_normalization in ["BatchNorm", "InstanceNorm"]:
            conv_block.append(
                self.layers_dict[self.network_normalization](num_features)
            )
        elif self.network_normalization is not None:
            raise ValueError(
                "The network normalization %s value must be in ['BatchNorm', 'InstanceNorm', None]"
                % self.network_normalization
            )
        return conv_block

    def return_layers_dict(self):
        if self.dimension == 3:
            layers = {
                "Conv": nn.Conv3d,
                "Pool": PadMaxPool3d,
                "InstanceNorm": nn.InstanceNorm3d,
                "BatchNorm": nn.BatchNorm3d,
            }
        elif self.dimension == 2:
            layers = {
                "Conv": nn.Conv2d,
                "Pool": PadMaxPool2d,
                "InstanceNorm": nn.InstanceNorm2d,
                "BatchNorm": nn.BatchNorm2d,
            }
        else:
            raise ValueError(
                "Cannot construct random network in dimension %i" % self.dimension
            )
        return layers

    @staticmethod
    def define_fc_layer(fc_dict, last_block=False):
        """
        Implement the FC block from the dictionnary fc_dict.

        Args:
            fc_dict: (dict) A dictionnary with the specifications to build a FC block
            - in_features (int) number of input neurones
            - out_features (int) number of output neurones
            last_block: (bool) indicates if the current FC layer is the last one of the network.
        Returns:
            (nn.Module) a list of modules in a nn.Sequential list
        """
        in_features = fc_dict["in_features"]
        out_features = fc_dict["out_features"]

        if last_block:
            fc_block = [nn.Linear(in_features, out_features)]
        else:
            fc_block = [nn.Linear(in_features, out_features), nn.LeakyReLU()]

        return nn.Sequential(*fc_block)

    def cascading_randomization(self, n, random_model=None):
        """
        Randomize then n last layers of the network.
        Similar as (Adebayo et al, 2018).

        Args:
            n: (int) number of layers to randomize
            random_model: (RandomArchitecture) random model to transfer identical weights
        Returns:
            self
        """
        fc_list = [
            ("classifier", "FC" + str(i)) for i in range(len(self.classifier) - 2)
        ]
        conv_list = [("features", "conv" + str(i)) for i in range(len(self.features))]
        layers_list = conv_list + fc_list
        if n > len(layers_list):
            raise ValueError(
                "The number of randomized layers %i cannot exceed the number of layers of the network %i"
                % (n, len(layers_list))
            )
        for i in range(-n, 0):
            block, name = layers_list[i]
            print(block, name)
            layer = getattr(getattr(self, block), name)

            # Independent or successive randomization
            if random_model is None:
                random_layer = deepcopy(layer)
                self.recursive_init(random_layer)
            else:
                random_layer = getattr(getattr(random_model, block), name)

            for j in range(len(layer)):
                layer[j] = random_layer[j]

        return self

    @staticmethod
    def recursive_init(layer):
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                RandomArchitecture.recursive_init(sub_layer)
        else:
            try:
                layer.reset_parameters()
            except AttributeError:
                pass

    def fix_first_layers(self, n):
        """
        Fix the n first model of the network.

        Args:
            n: (int) number of layers to fix
        Returns:
            self
        """
        fc_list = [
            ("classifier", "FC" + str(i)) for i in range(len(self.classifier) - 2)
        ]
        conv_list = [("features", "conv" + str(i)) for i in range(len(self.features))]
        layers_list = conv_list + fc_list
        if n > len(layers_list):
            raise ValueError(
                "The number of randomized layers %i cannot exceed the number of layers of the network %i"
                % (n, len(layers_list))
            )
        for i in range(n):
            block, name = layers_list[i]
            print(block, name)
            layer = getattr(getattr(self, block), name)
            for parameter in layer.parameters():
                parameter.requires_grad = False

        return self

    @staticmethod
    def fc_dict_design(n_fcblocks, convolutions, initial_shape, n_classes=2):
        """
        Sample parameters for a random architecture (FC part).

        Args:
            n_fcblocks: (int) number of fully connected blocks in the architecture.
            convolutions: (dict) parameters of the convolutional part.
            initial_shape: (array_like) shape of the initial input.
            n_classes: (int) number of classes in the classification problem.
        Returns:
            (dict) parameters of the architecture
            (list) the shape of the flattened layer
        """
        n_conv = len(convolutions)
        last_conv = convolutions["conv%i" % (len(convolutions) - 1)]
        out_channels = last_conv["out_channels"]
        flattened_shape = np.ceil(np.array(initial_shape) / 2 ** n_conv)
        flattened_shape[0] = out_channels
        in_features = np.product(flattened_shape)

        # Sample number of FC layers
        ratio = (in_features / n_classes) ** (1 / n_fcblocks)

        # Designing the parameters of each FC block
        fc = dict()
        for i in range(n_fcblocks):
            fc_dict = dict()
            out_features = in_features / ratio
            fc_dict["in_features"] = int(np.round(in_features))
            fc_dict["out_features"] = int(np.round(out_features))

            in_features = out_features
            fc["FC" + str(i)] = fc_dict

        return fc, flattened_shape