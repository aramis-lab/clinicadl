# coding: utf8

import os
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from .utils import load_model_after_ae, load_model_after_cnn
from .utils import train, test, patch_level_to_tsvs, soft_voting_to_tsvs

from ..tools.deep_learning import (EarlyStopping,
                                   save_checkpoint,
                                   create_model,
                                   load_model)
from ..tools.deep_learning.data import (MinMaxNormalization,
                                        load_data,
                                        MRIDataset_patch,
                                        MRIDataset_patch_hippocampus)


__author__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen", "Elina Thibeau-Sutre", "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"


def train_patch_single_cnn(params):

    model = create_model(params.model, params.gpu, dropout=params.dropout)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(
                params.tsv_path,
                params.diagnoses,
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline)

        print("Running for the %d-th fold" % fi)

        if params.transfer_learning_path is not None:
            if params.transfer_learning_autoencoder:
                print('Train the model with the weights from a pre-trained autoencoder.')
                model, _ = load_model_after_ae(
                        model,
                        os.path.join(
                            params.transfer_learning_path,
                            'best_model_dir',
                            "fold_" + str(fi),
                            'ConvAutoencoder',
                            'Encoder'),
                        filename='model_best_encoder.pth.tar')
            else:
                print('Train the model with the weights from a pre-trained CNN.')
                model, _ = load_model_after_cnn(
                        model,
                        os.path.join(
                            params.transfer_learning_path,
                            'best_model_dir',
                            "fold_" + str(fi),
                            'CNN',
                            params.selection),
                        filename='model_best.pth.tar')
        else:
            print('The model is trained from scratch.')
            model.load_state_dict(init_state)

        if params.hippocampus_roi:
            print("Only using hippocampus ROI")

            data_train = MRIDataset_patch_hippocampus(
                    params.input_dir,
                    training_tsv,
                    transformations=transformations
                    )
            data_valid = MRIDataset_patch_hippocampus(
                    params.input_dir,
                    valid_tsv,
                    transformations=transformations
                    )

        else:
            data_train = MRIDataset_patch(
                    params.input_dir,
                    training_tsv,
                    params.patch_size,
                    params.stride_size,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl
                    )
            data_valid = MRIDataset_patch(
                    params.input_dir,
                    valid_tsv,
                    params.patch_size,
                    params.stride_size,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl
                    )

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

        # Define loss and optimizer
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)

        loss = torch.nn.CrossEntropyLoss()

        print('Beginning the training task')
        # parameters used in training
        best_accuracy = 0.0
        best_loss_valid = np.inf

        writer_train_batch = SummaryWriter(
                log_dir=(
                    os.path.join(
                        params.output_dir,
                        "log_dir",
                        "fold_" + str(fi),
                        "CNN",
                        "train_batch"
                        )
                    )
                )

        writer_train_all_data = SummaryWriter(
                log_dir=(
                    os.path.join(
                        params.output_dir,
                        "log_dir",
                        "fold_" + str(fi),
                        "CNN",
                        "train_all_data"
                        )
                    )
                )

        writer_valid = SummaryWriter(
                log_dir=(
                    os.path.join(
                        params.output_dir,
                        "log_dir",
                        "fold_" + str(fi),
                        "CNN",
                        "valid"
                        )
                    )
                )

        # initialize the early stopping instance
        early_stopping = EarlyStopping(
                'min',
                min_delta=params.tolerance,
                patience=params.patience
                )

        for epoch in range(params.epochs):
            print("At %s-th epoch." % str(epoch))

            # train the model
            train_df, acc_mean_train, loss_batch_mean_train, global_step \
                = train(
                        model,
                        train_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_train_batch,
                        epoch,
                        model_mode='train',
                        selection_threshold=params.selection_threshold
                        )

            # calculate the training accuracy based on all the training data
            train_all_df, acc_mean_train_all, loss_batch_mean_train_all, _\
                = train(model, train_loader, params.gpu, loss, optimizer, writer_train_all_data, epoch,
                        model_mode='valid', selection_threshold=params.selection_threshold)
            print("For training, subject level balanced accuracy is %f at the end of epoch %d"
                  % (acc_mean_train_all, epoch))

            # at then end of each epoch, we validate one time for the model with the validation data
            valid_df, acc_mean_valid, loss_batch_mean_valid, _ \
                = train(
                        model,
                        valid_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_valid,
                        epoch,
                        model_mode='valid',
                        selection_threshold=params.selection_threshold
                        )

            print("For validation, subject level balanced accuracy is %f at the end of epoch %d"
                  % (acc_mean_valid, epoch))

            # save the best model based on the best loss and accuracy
            acc_is_best = acc_mean_valid > best_accuracy
            best_accuracy = max(best_accuracy, acc_mean_valid)
            loss_is_best = loss_batch_mean_valid < best_loss_valid
            best_loss_valid = min(loss_batch_mean_valid, best_loss_valid)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'loss': loss_batch_mean_valid,
                'accuracy': acc_mean_valid,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step},
                acc_is_best, loss_is_best,
                os.path.join(params.output_dir, "best_model_dir", "fold_" + str(fi), "CNN"))

            # try early stopping criterion
            if early_stopping.step(loss_batch_mean_valid) or epoch == params.epochs - 1:
                print("By applying early stopping or at the last epoch defined by user, "
                      "the training is stopped at %d-th epoch" % epoch)
                break

        # Final evaluation for all criteria
        for selection in ['best_loss', 'best_acc']:
            model, best_epoch = load_model(
                    model,
                    os.path.join(
                        params.output_dir,
                        'best_model_dir',
                        'fold_%i' % fi,
                        'CNN',
                        selection
                        ),
                    gpu=params.gpu,
                    filename='model_best.pth.tar')

            train_df, metrics_train = test(model, train_loader, params.gpu, loss)
            valid_df, metrics_valid = test(model, valid_loader, params.gpu, loss)

            # write the information of subjects and performances into tsv files.
            patch_level_to_tsvs(params.output_dir, train_df, metrics_train, fi,
                                dataset='train', selection=selection)
            patch_level_to_tsvs(params.output_dir, valid_df, metrics_valid, fi,
                                dataset='validation', selection=selection)

            soft_voting_to_tsvs(params.output_dir, fi, dataset='train', selection=selection,
                                selection_threshold=params.selection_threshold)
            soft_voting_to_tsvs(params.output_dir, fi, dataset='validation', selection=selection,
                                selection_threshold=params.selection_threshold)
            torch.cuda.empty_cache()
