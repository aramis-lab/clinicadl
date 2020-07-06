# coding: utf8

import os
import torch
from torch.utils.data import DataLoader

from ..tools.deep_learning.models import transfer_learning, init_model
from ..tools.deep_learning.data import (get_transforms,
                                        load_data,
                                        return_dataset,
                                        compute_num_cnn)
from ..tools.deep_learning.cnn_utils import train, soft_voting_to_tsvs
from clinicadl.test.test_multiCNN import test_cnn


def train_multi_cnn(params):
    """
    Trains one CNN per patch and writes for each CNN:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - final performances at the end of the training.
    Performances are also aggregated at the image level and combines the output of all networks.
    The initialization state is shared across all networks.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """

    transformations = get_transforms(params.mode, params.minmaxnormalization)

    num_cnn = compute_num_cnn(params.input_dir, params.tsv_path, params, data="train")

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        for cnn_index in range(num_cnn):

            training_df, valid_df = load_data(
                params.tsv_path,
                params.diagnoses,
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline)

            data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                        transformations, params, cnn_index=cnn_index)
            data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                        transformations, params, cnn_index=cnn_index)

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
            print('Initialization of the model %i' % cnn_index)
            model = init_model(params.model, gpu=params.gpu, dropout=params.dropout)
            model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                      gpu=params.gpu, selection=params.transfer_learning_selection)

            # Define criterion and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                                lr=params.learning_rate,
                                                                weight_decay=params.weight_decay)
            setattr(params, 'beginning_epoch', 0)

            # Define output directories
            log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs', "cnn-%i" % cnn_index,)
            model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models', "cnn-%i" % cnn_index)

            print('Beginning the training task')
            train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params)

            test_cnn(params.output_dir, train_loader, "train", fi, criterion, cnn_index, params, gpu=params.gpu)
            test_cnn(params.output_dir, valid_loader, "validation", fi, criterion, cnn_index, params, gpu=params.gpu)

        for selection in ['best_balanced_accuracy', 'best_loss']:
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                mode=params.mode,
                dataset='train',
                num_cnn=num_cnn,
                selection_threshold=params.selection_threshold)
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                mode=params.mode,
                dataset='validation',
                num_cnn=num_cnn,
                selection_threshold=params.selection_threshold)
