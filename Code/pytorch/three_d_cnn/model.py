from modules import *
import torch
"""
All the architectures are built here
"""


class Hosseini3(nn.Module):
    """
        Classifier for a 2-class classification task

        """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Hosseini3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3, stride=2),
            nn.ReLU(),

            nn.Conv3d(8, 8, 4, stride=2),
            nn.ReLU(),

            nn.Conv3d(8, 8, 3, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(8 * 14 * 17 * 14, 2000),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(2000, 500),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(500, n_classes)
        )

        self.flattened_shape = [-1, 8, 14, 17, 14]


class Hosseini2(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Hosseini2, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 4),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(8, 8, 4),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            nn.MaxPool3d(2, 2)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(8 * 14 * 17 * 14, 2000),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(2000, 500),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(500, n_classes)
        )

        self.flattened_shape = [-1, 8, 13, 16, 13]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Hosseini(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Hosseini, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(8 * 14 * 17 * 14, 2000),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(2000, 500),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(500, n_classes)
        )

        self.flattened_shape = [-1, 8, 14, 17, 14]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Esmaeilzadeh(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Esmaeilzadeh, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(3, 3),

            nn.Conv3d(64, 128, 3),
            nn.ReLU(),
            PadMaxPool3d(4, 4)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(128 * 5 * 6 * 5, 256),
            nn.ReLU(),

            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 128, 5, 6, 5]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def create_model(options):
    from classification_utils import load_model
    from os import path

    model = eval(options.model)()

    if options.gpu:  # TODO Check if version 0.4.1 allows loading a model saved on a different device
        model.cuda()
    else:
        model.cpu()

    if options.transfer_learning:
        model, _ = load_model(model, path.join(options.log_dir, "pretraining"), 'model_pretrained.pth.tar')
        for p in model.features.parameters():
            p.requires_grad = False

    return model


class Decoder(nn.Module):

    def __init__(self, model):
        from copy import deepcopy
        super(Decoder, self).__init__()

        self.encoder = deepcopy(model.features)
        self.decoder = self.construct_inv_layers(model)

        for i, layer in enumerate(self.encoder):
            if isinstance(layer, PadMaxPool3d):
                self.encoder[i].set_new_return()
            elif isinstance(layer, nn.MaxPool3d):
                self.encoder[i].return_indices = True

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

    def construct_inv_layers(self, model):
        inv_layers = []
        for layer in self.encoder:
            if isinstance(layer, nn.Conv3d):
                inv_layers.append(nn.ConvTranspose3d(layer.out_channels, layer.in_channels, layer.kernel_size,
                                                     stride=layer.stride))
            elif isinstance(layer, PadMaxPool3d):
                inv_layers.append(CropMaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.MaxPool3d):
                inv_layers.append(nn.MaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.Linear):
                inv_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, Flatten):
                inv_layers.append(Reshape(model.flattened_shape))
            elif isinstance(layer, nn.LeakyReLU):
                inv_layers.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
            else:
                inv_layers.append(layer)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)
