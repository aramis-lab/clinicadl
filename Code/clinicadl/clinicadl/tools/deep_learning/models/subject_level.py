from tools.deep_learning.modules import PadMaxPool3d, Flatten, CropMaxUnpool3d, Reshape
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


class Conv5_FC3_mni(nn.Module):
    """
    Classifier for a multi-class classification task

    Initially named Initial_architecture
    """
    def __init__(self, dropout=0.5):
        super(Conv5_FC3_mni, self).__init__()

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

            nn.Linear(128 * 4 * 5 * 4, 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, 2)

        )

        self.flattened_shape = [-1, 128, 4, 5, 4]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def transfer_from_autoencoder(experiment_path):
    import os

    # Find specific folders in experiment directory
    folds = os.listdir(os.path.join(experiment_path, "best_model_dir"))
    models = os.listdir(os.path.join(experiment_path, "best_model_dir", folds[0]))
    if models == ["ConvAutoencoder"]:
        return True
    return False


def create_model(options):
    from tools.deep_learning.classification_utils import load_model, check_and_clean
    from os import path
    import inspect

    choices = []
    for name, obj in inspect.getmembers:
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError('The model wanted %s has not been implemented in the module model.py' % options.model)
    model = eval(options.model)()

    if options.gpu:
        model.cuda()
    else:
        model.cpu()

    # TODO if resume must not check_and_clean
    # best_model_dir = path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split), 'CNN')
    # check_and_clean(best_model_dir)

    if options.transfer_learning is not None:
        if transfer_from_autoencoder(options.transfer_learning):
            print("A pretrained autoencoder is loaded at path %s" % options.transfer_learning)
            pretraining_path = path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split),
                                         'ConvAutoencoder', 'Encoder')
            apply_autoencoder_weights(model, options.transfer_learning, options)
            model, _ = load_model(model, pretraining_path, options.gpu, filename='model_pretrained.pth.tar')

        else:
            print("A pretrained model is loaded at path %s" % options.transfer_learning)
            apply_pretrained_network_weights(model, options.transfer_learning, options)
            pretraining_path = path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split), 'CNN')
            print("The model is saved at path", pretraining_path)
            model, _ = load_model(model, pretraining_path, options.gpu, filename='model_pretrained.pth.tar')

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
        inv_layers = self.replace_relu(inv_layers)
        inv_layers.reverse()
        return nn.Sequential(*inv_layers)

    @staticmethod
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


def apply_autoencoder_weights(model, experiment_path, options, difference=0):
    from copy import deepcopy
    import os
    from tools.deep_learning.classification_utils import save_checkpoint, check_and_clean

    decoder = Decoder(model)
    model_path = os.path.join(experiment_path, "best_model_dir", "fold_" + str(options.split), "ConvAutoencoder",
                              "best_loss", "model_best.pth.tar")
    initialize_other_autoencoder(decoder, model_path, difference=difference)

    model.features = deepcopy(decoder.encoder)
    for layer in model.features:
        if isinstance(layer, PadMaxPool3d):
            layer.set_new_return(False, False)

    pretraining_path = os.path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split),
                                    'ConvAutoencoder', 'Encoder')
    check_and_clean(pretraining_path)

    save_checkpoint({'model': model.state_dict(),
                     'epoch': -1,
                     'path': model_path},
                    False, False,
                    pretraining_path,
                    filename='model_pretrained.pth.tar')


def apply_pretrained_network_weights(model, experiment_path, options):
    import os
    from tools.deep_learning.classification_utils import save_checkpoint, check_and_clean

    model_path = os.path.join(experiment_path, "best_model_dir", "fold_" + str(options.split), "CNN",
                              "best_loss", "model_best.pth.tar")
    results = torch.load(model_path)
    model.load_state_dict(results['model'])

    pretraining_path = os.path.join(options.output_dir, 'best_model_dir', 'fold_' + str(options.split), 'CNN')
    check_and_clean(pretraining_path)

    save_checkpoint({'model': model.state_dict(),
                     'epoch': -1,
                     'path': model_path},
                    False, False,
                    pretraining_path,
                    filename='model_pretrained.pth.tar')


def initialize_other_autoencoder(decoder, pretrained_autoencoder_path, difference=0):

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

    return decoder


def correct_model_options(model_options):
    corrected_options = []
    for i, option in enumerate(model_options):
        if '-' not in option:
            new_option = '_'.join([corrected_options[-1], option])
            corrected_options[-1] = new_option
        else:
            corrected_options.append(option)

    return corrected_options



