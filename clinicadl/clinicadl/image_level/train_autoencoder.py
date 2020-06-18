# coding: utf8

from __future__ import print_function
from os import path
from time import time
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from warnings import warn

from ..tools.deep_learning.autoencoder_utils import train, visualize_image
from ..tools.deep_learning.data import MinMaxNormalization, MRIDataset, load_data
from ..tools.deep_learning import create_autoencoder, load_model


def train_autoencoder(params):
    """ Parameters
    params: class from utils module containing all the parameters for training a
    CNN.
    """

    if params.visualization:
        warn("The visualization of 3D image-based autoencoders was not implemented.")

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

    if params.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()
    criterion = torch.nn.MSELoss()

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fold in fold_iterator:

        training_tsv, valid_tsv = load_data(params.tsv_path, params.diagnoses,
                                            fold, params.n_splits,
                                            params.baseline)

        data_train = MRIDataset(params.input_dir, training_tsv,
                                params.preprocessing, transformations)
        data_valid = MRIDataset(params.input_dir, valid_tsv,
                                params.preprocessing, transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=params.batch_size,
                                  shuffle=True,
                                  num_workers=params.num_workers,
                                  pin_memory=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=params.num_workers,
                                  pin_memory=True
                                  )

        text_file = open(path.join(params.output_dir, 'python_version.txt'), 'w')
        text_file.write('Version of python: %s \n' % sys.version)
        text_file.write('Version of pytorch: %s \n' % torch.__version__)
        text_file.close()

        if params.transfer_learning_path is not None:
            transfer_learning_path = path.join(params.transfer_learning_path, "best_model_dir", "fold_%i" % fold,
                                               "ConvAutoencoder", "best_loss", "model_best.pth.tar")
        else:
            transfer_learning_path = None
        decoder = create_autoencoder(params.model, transfer_learning_path=transfer_learning_path,
                                     difference=params.transfer_learning_difference, gpu=params.gpu)
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)

        if params.add_sigmoid:
            if isinstance(decoder.decoder[-1], nn.ReLU):
                decoder.decoder = nn.Sequential(*list(decoder.decoder)[:-1])
            decoder.decoder.add_module("sigmoid", nn.Sigmoid())

        # Define output directories
        log_dir = path.join(params.output_dir, 'log_dir', 'fold_%i' % fold, 'ConvAutoencoder')
        visualization_dir = path.join(params.output_dir, 'autoencoder_reconstruction', 'fold_%i' % fold)
        model_dir = path.join(params.output_dir, 'best_model_dir', 'fold_%i' % fold, 'ConvAutoencoder')

        train(decoder, train_loader, valid_loader, criterion, optimizer, False,
              log_dir, model_dir, params)

        if params.visualization:
            print("Visualization of autoencoder reconstruction")
            best_decoder, _ = load_model(decoder, path.join(model_dir, "best_loss"),
                                         params.gpu, filename='model_best.pth.tar')
            visualize_image(best_decoder, valid_loader, path.join(visualization_dir, "validation"), nb_images=5)
            visualize_image(best_decoder, train_loader, path.join(visualization_dir, "train"), nb_images=5)
        del decoder
        torch.cuda.empty_cache()

    total_time = time() - total_time
    print('Total time', total_time)
