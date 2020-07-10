# coding: utf8
from os import path
import os
import torch
from torch import nn
from warnings import warn
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning.data import get_transforms, return_dataset, load_data, compute_num_cnn
from clinicadl.tools.deep_learning.models import transfer_learning, create_model, load_model, load_optimizer
from clinicadl.tools.deep_learning.cnn_utils import train, soft_voting_to_tsvs
from clinicadl.train.train_multiCNN import test_cnn
from . import file_len


def resume_multi_cnn(params):

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

    transformations = get_transforms(params.mode, params.minmaxnormalization)
    criterion = nn.CrossEntropyLoss()
    num_cnn = compute_num_cnn(params.input_dir, params.tsv_path, params)

    for fi in params.split:
        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline
        )

        ended_cnn = file_len(path.join(params.output_dir, 'fold-%i' % fi, '.ended'))

        for cnn_index in range(ended_cnn, num_cnn):

            print("Running for the %d-th fold" % fi)
            data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                        transformations, params, cnn_index=cnn_index)
            data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                        transformations, params, cnn_index=cnn_index)

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

            log_dir = path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs', 'cnn-%i' % cnn_index)
            model_dir = path.join(params.output_dir, 'fold-%i' % fi, 'models', 'cnn-%i' % cnn_index)

            model = create_model(params.model, dropout=params.dropout)

            # Check here that the checkpoint is present
            if path.exists(path.join(params.output_dir, 'fold-%i' % fi, 'models', 'cnn-%i' % cnn_index,
                           'checkpoint.pth.tar')):
                print('Initialization of the model %i' % cnn_index)
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

                test_cnn(params.output_dir, train_loader, "train", fi, criterion, cnn_index, params, gpu=params.gpu)
                test_cnn(params.output_dir, valid_loader, "validation", fi, criterion, cnn_index, params,
                         gpu=params.gpu)

                with open(os.path.join(params.output_dir, 'fold-%i' % fi, '.ended'), 'a') as ended_file:
                    ended_file.write("cnn-%i\n" % cnn_index)

            else:
                print('Initialization of the model %i' % cnn_index)
                model = transfer_learning(model, fi, source_path=params.transfer_learning_path,
                                          gpu=params.gpu, selection=params.transfer_learning_selection)

                optimizer = getattr(torch.optim, params.optimizer)(
                    filter(lambda x: x.requires_grad, model.parameters()),
                    lr=params.learning_rate,
                    weight_decay=params.weight_decay)
                setattr(params, 'beginning_epoch', 0)

                # Define output directories
                log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs', "cnn-%i" % cnn_index, )
                model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models', "cnn-%i" % cnn_index)

                print('Beginning the training task')
                train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params)

                test_cnn(params.output_dir, train_loader, "train", fi, criterion, cnn_index, params, gpu=params.gpu)
                test_cnn(params.output_dir, valid_loader, "validation", fi, criterion, cnn_index, params,
                         gpu=params.gpu)

                with open(os.path.join(params.output_dir, 'fold-%i' % fi, '.ended'), 'a') as ended_file:
                    ended_file.write("cnn-%i\n" % cnn_index)

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

        with open(os.path.join(params.output_dir, 'fold-%i' % fi, '.ended'), 'a') as ended_file:
            ended_file.write("final evaluation\n")
