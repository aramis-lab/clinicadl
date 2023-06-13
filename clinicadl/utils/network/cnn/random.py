import numpy as np

from clinicadl.utils.exceptions import ClinicaDLNetworksError
from clinicadl.utils.network.network_utils import *
from clinicadl.utils.network.sub_network import CNN


class RandomArchitecture(CNN):
    def __init__(
        self,
        convolutions_dict,
        n_fcblocks,
        input_size,
        dropout=0.5,
        network_normalization="BatchNorm",
        output_size=2,
        gpu=True,
    ):
        """
        Construct the Architecture randomly chosen for Random Search.

        Args:
            convolutions_dict: (dict) description of the convolutional blocks.
            n_fcblocks: (int) number of FC blocks in the network.
            input_size: (list) gives the structure of the input of the network.
            dropout: (float) rate of the dropout.
            network_normalization: (str) type of normalization layer in the network.
            output_size: (int) Number of output neurones of the network.
            gpu: (bool) If True the network weights are stored on a CPU, else GPU.
        """
        self.dimension = len(input_size) - 1
        self.first_in_channels = input_size[0]
        self.layers_dict = self.return_layers_dict()
        self.network_normalization = network_normalization

        convolutions = nn.Sequential()
        for key, item in convolutions_dict.items():
            convolutional_block = self.define_convolutional_block(item)
            convolutions.add_module(key, convolutional_block)

        classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))

        fc, flattened_shape = self.fc_dict_design(
            n_fcblocks, convolutions_dict, input_size, output_size
        )
        for key, item in fc.items():
            n_fc = int(key[2::])
            if n_fc == len(fc) - 1:
                fc_block = self.define_fc_layer(item, last_block=True)
            else:
                fc_block = self.define_fc_layer(item, last_block=False)
            classifier.add_module(key, fc_block)

        super().__init__(
            convolutions=convolutions,
            fc=classifier,
            n_classes=output_size,
            gpu=gpu,
        )

    def define_convolutional_block(self, conv_dict):
        """
        Design a convolutional block from the dictionary conv_dict.

        Args:
            conv_dict: (dict) A dictionary with the specifications to build a convolutional block
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
            raise ClinicaDLNetworksError(
                f"Dimension reduction {conv_dict['d_reduction']} is not supported. Please only include"
                "'MaxPooling' or 'stride' in your sampling options."
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
            raise ClinicaDLNetworksError(
                f"The network normalization {self.network_normalization} value must be in ['BatchNorm', 'InstanceNorm', None]"
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
                "Cannot construct random network in dimension {self.dimension}."
            )
        return layers

    @staticmethod
    def define_fc_layer(fc_dict, last_block=False):
        """
        Implement the FC block from the dictionary fc_dict.

        Args:
            fc_dict: (dict) A dictionary with the specifications to build a FC block
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
        last_conv = convolutions[f"conv{(len(convolutions) - 1)}"]
        out_channels = last_conv["out_channels"]
        flattened_shape = np.ceil(np.array(initial_shape) / 2**n_conv)
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
