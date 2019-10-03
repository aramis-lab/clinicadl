import argparse
import copy
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from .utils import load_model_after_ae, load_model_after_cnn
from .utils import MRIDataset_patch, train, test, patch_level_to_tsvs, soft_voting_to_tsvs

from ..tools.deep_learning.iotools import Parameters
from ..tools.deep_learning import EarlyStopping, save_checkpoint, commandline_to_json, create_model, load_model
from ..tools.deep_learning.data import MinMaxNormalization, load_data

__author__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen", "Elina Thibeau-Sutre", "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"


def train_patch_multi_cnn(params):

    model = create_model(params.model, params.gpu)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        for i in range(params.num_cnn):

            training_tsv, valid_tsv = load_data(params.tsv_path,
                    params.diagnoses,
                    fi,
                    n_splits=params.n_splits,
                    baseline=params.baseline)

            print("Running for the %d-th CNN" % i)
            if params.transfer_learning_path is not None:
              if params.transfer_learning_autoencoder:
                print('Train the model with the weights from a pre-trained autoencoder.')
                model_folder = os.path.join(params.transfer_learning_path,
                    'best_model_dir',
                    "fold_" + str(fi),
                    'ConvAutoencoder',
                    'Encoder')
                model, _ = load_model_after_ae(model,
                    model_folder,
                    filename='model_best_encoder.pth.tar')
              else:
                if params.transfer_learning_multicnn:
                  print('Train each of the models of multiple CNN with the weights from a pre-trained CNN.')
                  model_folder = os.path.join(params.transfer_learning_path,
                      'best_model_dir',
                      "fold_" + str(fi),
                      'cnn-' + str(i),
                      'best_acc')
                  model, _ = load_model_after_cnn(model,
                      model_folder,
                      filename='model_best.pth.tar')
                else:
                  print('Train the model with the weights from a pre-trained CNN.')
                  model_folder = os.path.join(params.transfer_learning_path,
                      'best_model_dir',
                      "fold_" + str(fi),
                      'CNN',
                      'best_acc')
                  model, _ = load_model_after_cnn(model,
                      model_folder,
                      filename='model_best.pth.tar')
            else:
              print('The model is trained from scratch.')
              model.load_state_dict(init_state)

            data_train = MRIDataset_patch(params.input_dir,
                    training_tsv,
                    params.patch_size,
                    params.patch_stride,
                    transformations=transformations,
                    patch_index=i,
                    prepare_dl=params.prepare_dl)
            data_valid = MRIDataset_patch(params.input_dir,
                    valid_tsv,
                    params.patch_size,
                    params.patch_stride,
                    transformations=transformations,
                    patch_index=i,
                    prepare_dl=params.prepare_dl)

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

            # Define loss and optimizer
            optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                                 params.learning_rate, weight_decay=params.weight_decay)
            loss = torch.nn.CrossEntropyLoss()

            print('Beginning the training task')
            # parameters used in training
            best_accuracy = 0.0
            best_loss_valid = np.inf
            writer_train_batch = SummaryWriter(log_dir=(os.path.join(params.output_dir, "log_dir", "fold_%i" % fi,
                                                                     "cnn-%i" % i, "train_batch")))
            writer_train_all_data = SummaryWriter(log_dir=(os.path.join(params.output_dir, "log_dir", "fold_%i" % fi,
                                                                        "cnn-%i" % i, "train_all_data")))
            writer_valid = SummaryWriter(log_dir=(os.path.join(params.output_dir, "log_dir", "fold_%i" % fi,
                                                               "cnn-%i" % i, "valid")))

            # initialize the early stopping instance
            early_stopping = EarlyStopping('min', min_delta=params.tolerance, patience=params.patience)

            for epoch in range(params.epochs):
                print("At %i-th epoch." % epoch)

                # train the model
                train_df, acc_mean_train, loss_batch_mean_train, global_step,\
                    = train(model,
                        train_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_train_batch,
                        epoch,
                        model_mode='train')

                # calculate the training accuracy based on all the training data
                train_all_df, acc_mean_train_all, loss_batch_mean_train_all, _,\
                    = train(model,
                        train_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_train_all_data,
                        epoch,
                        model_mode='valid')
                print("For training, subject level balanced accuracy is %f at the end of epoch %d" % (acc_mean_train_all, epoch))

                # at then end of each epoch, we validate one time for the model
                # with the validation data
                
                valid_df, acc_mean_valid, loss_batch_mean_valid, _\
                    = train(model,
                        valid_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_valid,
                        epoch,
                         model_mode='valid')
                print("For validation, subject level balanced accuracy is %f at the end of epoch %d" % (acc_mean_valid, epoch))

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
                    os.path.join(params.output_dir,
                      "best_model_dir",
                      "fold_%i" % fi,
                      "cnn-%i" % i))

                # try early stopping criterion
                if early_stopping.step(loss_batch_mean_valid) or epoch == params.epochs - 1:
                    print("By applying early stopping or at the last epoch defined by user,"
                          "the training is stopped at %d-th epoch" % epoch)

                    break

            for selection in ['best_acc', 'best_loss']:
                # load the best trained model during the training
                model, best_epoch = load_model(model,
                    os.path.join(params.output_dir,
                      'best_model_dir',
                      'fold_%i' % fi,
                      'cnn-%i' % i,
                      selection),
                    gpu=params.gpu,
                    filename='model_best.pth.tar')

                train_df, metrics_train = test(model,
                    train_loader,
                    params.gpu,
                    loss)
                valid_df, metrics_valid = test(model,
                    valid_loader,
                    params.gpu,
                    loss)
                patch_level_to_tsvs(params.output_dir,
                    train_df, metrics_train, fi,
                    selection, dataset='train', cnn_index=i)
                patch_level_to_tsvs(params.output_dir,
                    valid_df, metrics_valid, fi,
                    selection, dataset='validation', cnn_index=i)

                torch.cuda.empty_cache()

        for selection in ['best_acc', 'best_loss']:
          soft_voting_to_tsvs(params.output_dir, 
              fi,
              selection,
              dataset='train',
              num_cnn=params.num_cnn,
              selection_threshold=params.selection_threshold)
          soft_voting_to_tsvs(params.output_dir,
              fi,
              selection,
              dataset='validation',
              num_cnn=params.num_cnn,
              selection_threshold=params.selection_threshold)
