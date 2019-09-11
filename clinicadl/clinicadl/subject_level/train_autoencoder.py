from __future__ import print_function
import argparse
from os import path
from time import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import ae_finetuning
from tools.deep_learning import Parameters
from tools.deep_learning.data import MinMaxNormalization, MRIDataset, load_data
from tools.deep_learning import create_autoencoder, commandline_to_json


def train_autoencoder(params):
""" Parameters
params: class from utils module containing all the parameters for training a 
CNN.
"""

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

    if params.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()
    criterion = torch.nn.MSELoss()

    training_tsv, valid_tsv = load_data(params.tsv_path, params.diagnoses,
                                        params.split, params.n_splits, 
                                        params.baseline)

    data_train = MRIDataset(params.input_dir, training_tsv, 
                            params.preprocessing, transformations)
    data_valid = MRIDataset(params.input_dir, valid_tsv, 
                            params.preprocessing, transformations)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              params.batch_size,
                              shuffle=True,
                              params.num_workers,
                              drop_last=True
                              )

    valid_loader = DataLoader(data_valid,
                              )

    valid_loader = DataLoader(data_valid,
                              batch_size=params.batch_size,
                              shuffle=False,
                              num_workers=params.num_workers,
                              drop_last=False
                              )

    text_file = open(path.join(params.output_dir, 'python_version.txt'), 'w')
    text_file.write('Version of python: %s \n' % sys.version)
    text_file.write('Version of pytorch: %s \n' % torch.__version__)
    text_file.close()

    decoder = create_autoencoder(params.model, params.pretrained_path, 
                                 difference=params.pretrained_difference)
    optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                         params.learning_rate, params.weight_decay)

    if params.add_sigmoid:
        if isinstance(decoder.decoder[-1], nn.ReLU):
            decoder.decoder = nn.Sequential(*list(decoder.decoder)[:-1])
        decoder.decoder.add_module("sigmoid", nn.Sigmoid())

    ae_finetuning(decoder, train_loader, valid_loader, criterion, params.optimizer, False, params)

    total_time = time() - total_time
    print('Total time', total_time)


#if __name__ == "__main__":
#    commandline = parser.parse_known_args()
#    commandline_to_json(commandline, 'ConvAutoencoder')
#    options = commandline[0]
#    if commandline[1]:
#        print("unknown arguments: %s" % parser.parse_known_args()[1])
#    train_params_autoencoder = Parameters(tsv_path, output_dir, input_dir, model)
#    train_params_autoencoder.write(options)
#    train_autoencoder(train_parameters_autoencoder)
