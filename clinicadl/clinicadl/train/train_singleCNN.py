# coding: utf8

import os
import torch
from torch.utils.data import DataLoader

from ..tools.deep_learning.models import transfer_learning, init_model, create_model, load_model
from ..tools.deep_learning.data import (get_transforms,
                                        load_data,
                                        return_dataset)
from ..tools.deep_learning.cnn_utils import train, test, mode_level_to_tsvs, soft_voting_to_tsvs
from ..tools.deep_learning.iotools import return_logger


def train_single_cnn(params):
    """
    Trains a single CNN and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - final performances at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """

    main_logger = return_logger(params.verbosity, "main process")
    train_logger = return_logger(params.verbosity, "train")
    eval_logger = return_logger(params.verbosity, "final evaluation")
    transformations = get_transforms(params.mode, params.minmaxnormalization)

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:
        main_logger.info("Fold %i" % fi)

        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline,
            logger=main_logger
        )

        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    transformations, params)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    transformations, params)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(
            data_train,
            batch_size=params.batch_size,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True
        )

        valid_loader = DataLoader(
            data_valid,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True
        )

        # Initialize the model
        main_logger.info('Initialization of the model')
        model = init_model(params.model, gpu=params.gpu, dropout=params.dropout)
        model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                  gpu=params.gpu, selection=params.transfer_learning_selection,
                                  logger=main_logger)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'models')

        main_logger.debug('Beginning the training task')
        train(model, train_loader, valid_loader, criterion,
              optimizer, False, log_dir, model_dir, params, train_logger)

        test_single_cnn(model, params.output_dir, train_loader, "train",
                        fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu)
        test_single_cnn(model, params.output_dir, valid_loader, "validation",
                        fi, criterion, params.mode, eval_logger, params.selection_threshold, gpu=params.gpu)


def test_single_cnn(model, output_dir, data_loader, subset_name, split, criterion, mode, logger, selection_threshold,
                    gpu=False):

    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(model, os.path.join(output_dir, 'fold-%i' % split, 'models', selection),
                                       gpu=gpu, filename='model_best.pth.tar')

        results_df, metrics = test(model, data_loader, gpu, criterion, mode)
        logger.info("%s level %s balanced accuracy is %f for model selected on %s"
                    % (mode, subset_name, metrics["balanced_accuracy"], selection))

        mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, mode, dataset=subset_name)

        # Soft voting
        if mode in ["patch", "roi", "slice"]:
            soft_voting_to_tsvs(output_dir, split, logger=logger, selection=selection, mode=mode,
                                dataset=subset_name, selection_threshold=selection_threshold)
