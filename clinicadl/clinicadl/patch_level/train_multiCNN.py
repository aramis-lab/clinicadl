# coding: utf8

import copy
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ..tools.deep_learning.models.autoencoder import transfer_learning
from ..tools.deep_learning import create_model
from ..tools.deep_learning.data import (MinMaxNormalization,
                                        load_data,
                                        MRIDataset_patch)
from ..tools.deep_learning.cnn_utils import train, soft_voting_to_tsvs
from .evaluation_multiCNN import test_cnn


__author__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen", "Elina Thibeau-Sutre", "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"


def train_patch_multi_cnn(params):

    model = create_model(params.model, params.gpu, dropout=params.dropout)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        for cnn_index in range(params.num_cnn):

            training_tsv, valid_tsv = load_data(
                    params.tsv_path,
                    params.diagnoses,
                    fi,
                    n_splits=params.n_splits,
                    baseline=params.baseline)

            print("Running for the %d-th CNN" % cnn_index)
            model.load_state_dict(init_state)
            model = transfer_learning(model, fi, transfer_learning_autoencoder=params.transfer_learning_autoencoder,
                                      source_path=params.transfer_learning_path, gpu=params.gpu,
                                      selection=params.transfer_learning_selection, cnn_index=cnn_index)

            data_train = MRIDataset_patch(
                    params.input_dir,
                    training_tsv,
                    params.patch_size,
                    params.stride_size,
                    preprocessing=params.preprocessing,
                    transformations=transformations,
                    patch_index=cnn_index,
                    prepare_dl=params.prepare_dl
                    )

            data_valid = MRIDataset_patch(
                    params.input_dir,
                    valid_tsv,
                    params.patch_size,
                    params.stride_size,
                    preprocessing=params.preprocessing,
                    transformations=transformations,
                    patch_index=cnn_index,
                    prepare_dl=params.prepare_dl
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
            model.load_state_dict(init_state)
            model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                      transfer_learning_autoencoder=params.transfer_learning_autoencoder,
                                      gpu=params.gpu, selection=params.transfer_learning_selection)

            # Define criterion and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                                lr=params.learning_rate,
                                                                weight_decay=params.weight_decay)
            setattr(params, 'beginning_epoch', 0)

            # Define output directories
            log_dir = os.path.join(params.output_dir, "log_dir", "fold_%i" % fi, "cnn-%i" % cnn_index,)
            model_dir = os.path.join(params.output_dir, "best_model_dir", "fold_%i" % fi, "cnn-%i" % cnn_index)

            print('Beginning the training task')
            train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params)

            test_cnn(train_loader, "train", fi, criterion, cnn_index, params)
            test_cnn(valid_loader, "validation", fi, criterion, cnn_index, params)

        for selection in ['best_acc', 'best_loss']:
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                mode=params.mode,
                dataset='train',
                num_cnn=params.num_cnn,
                selection_threshold=params.selection_threshold)
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                mode=params.mode,
                dataset='validation',
                num_cnn=params.num_cnn,
                selection_threshold=params.selection_threshold)
