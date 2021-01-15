# coding: utf8

from torch import nn
import torch
from copy import deepcopy

from .modules import PadMaxPool3d, PadMaxPool2d, CropMaxUnpool3d, CropMaxUnpool2d, Flatten, Reshape


class AutoEncoder(nn.Module):

    def __init__(self, model=None):
        """
        Construct an autoencoder from a given CNN. The encoder part corresponds to the convolutional part of the CNN.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        """
        from copy import deepcopy
        super(AutoEncoder, self).__init__()

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
                inv_layers.append(nn.ConvTranspose3d(layer.out_channels, layer.in_channels, layer.kernel_size,
                                                     stride=layer.stride, padding=layer.padding))
                self.level += 1
            elif isinstance(layer, PadMaxPool3d):
                inv_layers.append(CropMaxUnpool3d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, PadMaxPool2d):
                inv_layers.append(CropMaxUnpool2d(layer.kernel_size, stride=layer.stride))
            elif isinstance(layer, nn.Linear):
                inv_layers.append(nn.Linear(layer.out_features, layer.in_features))
            elif isinstance(layer, Flatten):
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
                inv_layers[idx_relu], inv_layers[idx_conv] = inv_layers[idx_conv], inv_layers[idx_relu]
                idx_conv, idx_relu = -1, -1

        # Check if number of features of batch normalization layers is still correct
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.BatchNorm3d):
                conv = inv_layers[idx + 1]
                inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)

        return inv_layers


def transfer_learning(model, split, source_path=None, gpu=False,
                      selection="best_balanced_accuracy", cnn_index=None,
                      logger=None):
    """
    Allows transfer learning from a CNN or an autoencoder to a CNN

    :param model: (nn.Module) the target CNN of the transfer learning.
    :param split: (int) the fold number (for serialization purpose).
    :param source_path: (str) path to the source experiment.
    :param gpu: (bool) If True a GPU is used.
    :param selection: (str) chooses on which criterion the source model is selected (ex: best_loss, best_acc)
    :param cnn_index: (int) index of the CNN to be loaded (if transfer from a multi-CNN).
    :return: (nn.Module) the model after transfer learning.
    """
    import argparse
    from os import path
    from ..iotools import read_json, translate_parameters
    import logging

    if logger is None:
        logger = logging

    if source_path is not None:
        source_commandline = argparse.Namespace()
        source_commandline = read_json(source_commandline, json_path=path.join(source_path, "commandline.json"))
        source_commandline = translate_parameters(source_commandline)
        if source_commandline.mode_task == "autoencoder":
            logger.info("A pretrained autoencoder is loaded at path %s" % source_path)
            model = transfer_autoencoder_weights(model, source_path, split)

        else:
            logger.info("A pretrained CNN is loaded at path %s" % source_path)
            model = transfer_cnn_weights(model, source_path, split, selection=selection, cnn_index=cnn_index)

    else:
        logger.info("The model is trained from scratch.")

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def transfer_autoencoder_weights(model, source_path, split):
    """
    Set the weights of the model according to the autoencoder at source path.
    The encoder part of the autoencoder must exactly correspond to the convolutional part of the model.

    :param model: (Module) the model which must be initialized
    :param source_path: (str) path to the source task experiment
    :param split: (int) split number to load
    :return: (str) path to the written weights ready to be loaded
    """
    from copy import deepcopy
    import os

    if not isinstance(model, AutoEncoder):
        decoder = AutoEncoder(model)
    else:
        decoder = model

    model_path = os.path.join(source_path, 'fold-%i' % split, 'models', "best_loss", "model_best.pth.tar")
    source_dict = torch.load(model_path)

    initialize_other_autoencoder(decoder, source_dict)

    if not isinstance(model, AutoEncoder):
        model.features = deepcopy(decoder.encoder)
        for layer in model.features:
            if isinstance(layer, PadMaxPool3d):
                layer.set_new_return(False, False)

    return model


def transfer_cnn_weights(model, source_path, split, selection="best_balanced_accuracy", cnn_index=None):
    """
    Set the weights of the model according to the CNN at source path.
    :param model: (Module) the model which must be initialized
    :param source_path: (str) path to the source task experiment
    :param split: (int) split number to load
    :param selection: (str) chooses on which criterion the source model is selected (ex: best_loss, best_acc)
    :param cnn_index: (int) index of the CNN to be loaded (if transfer from a multi-CNN).
    :return: (str) path to the written weights ready to be loaded
    """

    import os
    import torch

    if isinstance(model, AutoEncoder):
        raise ValueError('Transfer learning from CNN to autoencoder was not implemented.')

    model_path = os.path.join(source_path, "fold-%i" % split, "models", selection, "model_best.pth.tar")
    if cnn_index is not None and not os.path.exists(model_path):
        print("Transfer learning from multi-CNN, cnn-%i" % cnn_index)
        model_path = os.path.join(source_path, "fold_%i" % split, "models", "cnn-%i" % cnn_index,
                                  selection, "model_best.pth.tar")
    results = torch.load(model_path)
    model.load_state_dict(results['model'])

    return model


def initialize_other_autoencoder(decoder, source_dict):
    """
    Initialize an autoencoder with another one values even if they have different sizes.
    :param decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
    :param source_dict: (dict) The result dict produced by save_checkpoint.
    :return: (Autoencoder) initialized autoencoder
    """

    try:
        decoder.load_state_dict(source_dict['model'])
    except RuntimeError:
        print("The source and target autoencoders do not have the same size."
              "The transfer learning task may not work correctly for custom models.")

        parameters_dict = source_dict['model']
        difference = find_maximum_layer(decoder.state_dict()) - find_maximum_layer(parameters_dict)

        for key in parameters_dict.keys():
            section, number, spec = key.split('.')
            number = int(number)
            if section == 'encoder' and number < len(decoder.encoder):
                data = getattr(getattr(decoder, section)[number], spec).data
                assert data.shape == parameters_dict[key].shape
                getattr(getattr(decoder, section)[number], spec).data = parameters_dict[key]
            elif section == 'decoder':
                # Deeper target autoencoder
                if difference >= 0:
                    data = getattr(getattr(decoder, section)[number + difference], spec).data
                    assert data.shape == parameters_dict[key].shape
                    getattr(getattr(decoder, section)[number + difference], spec).data = parameters_dict[key]
                # More shallow target autoencoder
                elif difference < 0 and number < len(decoder.decoder):
                    data = getattr(getattr(decoder, section)[number], spec).data
                    new_key = '.'.join(['decoder', str(number + abs(difference)), spec])
                    assert data.shape == parameters_dict[new_key].shape
                    getattr(getattr(decoder, section)[number], spec).data = parameters_dict[new_key]

    return decoder


def find_maximum_layer(state_dict):
    max_layer = 0
    for key in state_dict.keys():
        _, num, _ = key.split(".")
        num = int(num)
        if num > max_layer:
            max_layer = num
    return max_layer
