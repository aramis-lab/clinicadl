from modules import *
import torch.nn as nn
import torch
from copy import deepcopy

"""
All the architectures are built here
"""


class Test2_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test2_batch_drop(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2_batch_drop, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Dropout(p=0.5),
            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x



class Test2(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 5000),
            nn.ReLU(),

            nn.Linear(5000, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test3_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test3_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test3(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 500),
            nn.ReLU(),

            nn.Linear(500, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test4_batch(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test4_batch, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test4(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Test4, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2)
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Optim(nn.Module):
    """
    Classifier for a multi-class classification task
    """
    def __init__(self):
        super(Optim, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            Flatten(),

            nn.Linear(16 * 41 * 51 * 44, 100),
            nn.ReLU(),

            nn.Linear(100, 2)

        )

        self.flattened_shape = [-1, 16, 41, 51, 44]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Rieke(nn.Module):
    """
    Classifier for a multi-class classification task

    """
    def __init__(self, dropout=0.0, n_classes=2):
        super(Rieke, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
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
            nn.Linear(64 * 4 * 5 * 4, 128),
            nn.ReLU(),

            nn.Linear(128, n_classes)
        )

        self.flattened_shape = [-1, 64, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

    def __len__(self):
        return len(self.layers)


class Test(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Test, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32)
        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 4 * 5 * 4, 256),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 32, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Test_nobatch(nn.Module):
    """
    Classifier for a 2-class classification task

    """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Test_nobatch, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
        )

        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 5 * 6 * 5, 256),
            nn.ReLU(),

            nn.Dropout(p=0.0),
            nn.Linear(256, n_classes)
        )

        self.flattened_shape = [-1, 32, 5, 6, 5]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_3(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_3, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(32 * 20 * 25 * 21, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 32, 20, 25, 21]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_4(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_4, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

        )
        self.classifier = nn.Sequential(
            # Fully connected layers
            Flatten(),

            nn.Dropout(p=dropout),
            nn.Linear(64 * 9 * 12 * 10, 1000),
            nn.ReLU(),

            nn.Linear(1000, n_classes)
        )

        self.flattened_shape = [-1, 64, 9, 12, 10]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


class Conv_5(nn.Module):
    """
       Classifier for a 2-class classification task

       """

    def __init__(self, dropout=0.0, n_classes=2):
        super(Conv_5, self).__init__()

        self.features = nn.Sequential(
            # Convolutions
            nn.Conv3d(1, 16, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 32, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 64, 3),
            nn.ReLU(),
            PadMaxPool3d(2, 2),
            nn.BatchNorm3d(64),

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


def create_model(options):
    from classification_utils import load_model
    from os import path

    model = eval(options.model)()

    if options.gpu:  # TODO Check if version 0.4.1 allows loading a model saved on a different device
        model.cuda()
    else:
        model.cpu()

    if options.transfer_learning is not None:
        model, _ = load_model(model, path.join(options.log_dir, "pretraining"), 'model_pretrained.pth.tar')

    return model


class Decoder(nn.Module):

    def __init__(self, model=None):
        from copy import deepcopy
        super(Decoder, self).__init__()

        self.level = 0

        if model is not None:
            self.encoder = deepcopy(model.features)
            self.decoder = self.construct_inv_layers(model)

            for i, layer in enumerate(self.encoder):
                if isinstance(layer, PadMaxPool3d):
                    self.encoder[i].set_new_return()
                elif isinstance(layer, nn.MaxPool3d):
                    self.encoder[i].return_indices = True
        else:
            self.encoder = nn.Sequential()
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

    def construct_inv_layers(self, model):
        inv_layers = []
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Conv3d):
                inv_layers.append(nn.ConvTranspose3d(layer.out_channels, layer.in_channels, layer.kernel_size,
                                                     stride=layer.stride))
                self.level += 1
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
            elif i == len(self.encoder) - 1 and isinstance(layer, nn.BatchNorm3d):
                pass
            else:
                inv_layers.append(layer)
        inv_layers = replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)


def apply_autoencoder_weights(model, pretrained_autoencoder_path, model_path):
    from copy import deepcopy
    from os import path
    import os
    from classification_utils import save_checkpoint

    decoder = Decoder(model)
    result_dict = torch.load(pretrained_autoencoder_path)

    decoder.load_state_dict(result_dict['model'])
    model.features = deepcopy(decoder.encoder)
    if not path.exists(path.join(model_path, 'pretraining')):
        os.makedirs(path.join(model_path, "pretraining"))

    save_checkpoint({'model': model.state_dict(),
                     'epoch': -1,
                     'path': pretrained_autoencoder_path},
                    False,
                    path.join(model_path, "pretraining"),
                    'model_pretrained.pth.tar')


def replace_relu(inv_layers):
    idx_relu, idx_conv = -1, -1
    for idx, layer in enumerate(inv_layers):
        if isinstance(layer, nn.ConvTranspose3d):
            idx_conv = idx
        elif isinstance(layer, nn.ReLU):
            idx_relu = idx

        if idx_conv != -1 and idx_relu != -1:
            inv_layers[idx_relu], inv_layers[idx_conv] = inv_layers[idx_conv], inv_layers[idx_relu]
            idx_conv, idx_relu = -1, -1

    # Check if number of features of batch normalization layers is still correct
    for idx, layer in enumerate(inv_layers):
        if isinstance(layer, nn.BatchNorm3d):
            conv = inv_layers[idx + 1]
            inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)

    return inv_layers

