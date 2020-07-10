# coding: utf8
from os import path
import os
from torch import nn
from warnings import warn
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning.data import get_transforms, return_dataset, load_data
from clinicadl.tools.deep_learning import create_model, load_model, load_optimizer
from clinicadl.tools.deep_learning.cnn_utils import train
from clinicadl.test.test_singleCNN import test_cnn


def resume_single_cnn(params):

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

    transformations = get_transforms(params.mode, params.minmaxnormalization)
    criterion = nn.CrossEntropyLoss()

    for fi in params.split:
        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline
        )

        print("Running for the %d-th fold" % fi)
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
            pin_memory=True)

        valid_loader = DataLoader(
            data_valid,
            batch_size=params.batch_size,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True)

        log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models')

        model = create_model(params.model, dropout=params.dropout)
        model, current_epoch, num_bad_epochs = load_model(
            model,
            model_dir,
            gpu=params.gpu,
            filename='checkpoint.pth.tar')
        optimizer_path = path.join(model_dir, 'optimizer.pth.tar')
        optimizer = load_optimizer(optimizer_path, model)

        # Prepare args for resuming
        params.beginning_epoch = current_epoch + 1
        if num_bad_epochs is None:
            warn("You are resuming a job conducted with an older version of clinicadl."
                 "Information on patience for early stopping was lost.")
            params.num_bad_epochs = 0
        else:
            params.num_bad_epochs = num_bad_epochs

        print('Resuming the training task')
        train(model, train_loader, valid_loader, criterion, optimizer, True,
              log_dir, model_dir, params)

        test_cnn(params.output_dir, train_loader, "train",
                 fi, criterion, params, gpu=params.gpu)
        test_cnn(params.output_dir, valid_loader, "validation",
                 fi, criterion, params, gpu=params.gpu)
        ended_file = open(os.path.join(params.output_dir, "fold-%i" % fi, ".ended"), 'w')
        ended_file.close()
