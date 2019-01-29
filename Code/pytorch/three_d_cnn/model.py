from modules import *
import torch.nn as nn
import torch
from copy import deepcopy

"""
All the architectures are built here
"""


class Conv5_FC3(nn.Module):
    """
    Classifier for a multi-class classification task

    Initially named Initial_architecture
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            PadMaxPool3d(2, 2),

        )

        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(128 * 6 * 7 * 6, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 6, 7, 6]

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
                                                     stride=layer.stride, padding=layer.padding))
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
                inv_layers.append(deepcopy(layer))
        inv_layers = replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)


def apply_autoencoder_weights(model, pretrained_autoencoder_path, model_path, difference=0):
    from copy import deepcopy
    from os import path
    import os
    from classification_utils import save_checkpoint

    decoder = Decoder(model)
    initialize_other_autoencoder(decoder, pretrained_autoencoder_path, model_path, difference=difference)

    model.features = deepcopy(decoder.encoder)
    if not path.exists(path.join(model_path, 'pretraining')):
        os.makedirs(path.join(model_path, "pretraining"))

    save_checkpoint({'model': model.state_dict(),
                     'epoch': -1,
                     'path': pretrained_autoencoder_path},
                    False, False,
                    path.join(model_path, "pretraining"),
                    filename='model_pretrained.pth.tar')


def initialize_other_autoencoder(decoder, pretrained_autoencoder_path, model_path, difference=0):
    from os import path
    import os
    from classification_utils import save_checkpoint

    result_dict = torch.load(pretrained_autoencoder_path)
    parameters_dict = result_dict['model']
    module_length = int(len(decoder) / decoder.level)
    difference = difference * module_length

    for key in parameters_dict.keys():
        section, number, spec = key.split('.')
        number = int(number)
        if section == 'encoder' and number < len(decoder.encoder):
            data_ptr = eval('decoder.' + section + '[number].' + spec + '.data')
            data_ptr = parameters_dict[key]
        elif section == 'decoder':
            # Deeper autoencoder
            if difference >= 0:
                data_ptr = eval('decoder.' + section + '[number + difference].' + spec + '.data')
                data_ptr = parameters_dict[key]
            # More shallow autoencoder
            elif difference < 0 and number < len(decoder.decoder):
                data_ptr = eval('decoder.' + section + '[number].' + spec + '.data')
                new_key = '.'.join(['decoder', str(number + difference), spec])
                data_ptr = parameters_dict[new_key]

    if not path.exists(path.join(model_path, 'pretraining')):
        os.makedirs(path.join(model_path, "pretraining"))

    save_checkpoint({'model': decoder.state_dict(),
                     'epoch': -1,
                     'path': pretrained_autoencoder_path},
                    False, False,
                    path.join(model_path, "pretraining"),
                    'model_pretrained.pth.tar')
    return decoder


def replace_relu(inv_layers):
    idx_relu, idx_conv = -1, -1
    for idx, layer in enumerate(inv_layers):
        if isinstance(layer, nn.ConvTranspose3d):
            idx_conv = idx
        elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
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

