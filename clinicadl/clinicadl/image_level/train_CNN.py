# coding: utf8

import torch
import sys
from time import time
from os import path
from torch.utils.data import DataLoader

from .utils import train
from .evaluation import test_cnn
from ..tools.deep_learning.data import MinMaxNormalization, MRIDataset, load_data
from ..tools.deep_learning import create_model
from ..tools.deep_learning.models import transfer_learning


def train_cnn(params):
    """
    This function trains a cnn network using the subject-level features

    Parameters
    ----------

    params : class clinicadl.tools.deep_learning.iotools.Parameters
      Structure containing the parameters/options to launch the trainning of a CNN network.

    Returns
    _______

    Models are stored in disk.

    """

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

    if params.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()

    text_file = open(path.join(params.output_dir, 'python_version.txt'), 'w')
    text_file.write('Version of python: %s \n' % sys.version)
    text_file.write('Version of pytorch: %s \n' % torch.__version__)
    text_file.close()

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fold in fold_iterator:

        # Get the data.
        training_tsv, valid_tsv = load_data(params.tsv_path, params.diagnoses,
                                            fold, params.n_splits,
                                            params.baseline)

        data_train = MRIDataset(
            params.input_dir,
            training_tsv,
            params.preprocessing,
            transform=transformations
        )
        data_valid = MRIDataset(
            params.input_dir,
            valid_tsv,
            params.preprocessing,
            transform=transformations
        )

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

        # Initialize the model
        print('Initialization of the model')
        model = create_model(params.model, params.gpu, dropout=params.dropout)
        model = transfer_learning(model, fold, source_path=params.transfer_learning_path,
                                  transfer_learning_autoencoder=params.transfer_learning_autoencoder,
                                  gpu=params.gpu, selection=params.transfer_learning_selection)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = path.join(params.output_dir, 'log_dir', 'fold_%i' % fold, 'CNN')
        model_dir = path.join(params.output_dir, 'best_model_dir', 'fold_%i' % fold, 'CNN')

        train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params)

        params.model_path = params.output_dir
        test_cnn(train_loader, "train", fold, criterion, params)
        test_cnn(valid_loader, "validation", fold, criterion, params)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
