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

from tools.deep_learning import EarlyStopping, save_checkpoint, commandline_to_json, create_model, load_model
from tools.deep_learning.data import MinMaxNormalization, load_data

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch CNN with multiCNN")

# Mandatory arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to tsv file of the population based on the diagnosis tsv files."
                         "To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs and the tsv files containing the performances.")

# Data management
parser.add_argument("--diagnoses", default=["sMCI", "pMCI"], type=str, nargs="+",
                    help="Labels based on binary classification")
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument('--prepare_dl', default=False, action="store_true",
                    help="If True the outputs of preprocessing prepare_dl are used, else the whole MRI is loaded.")

# Transfer learning
parser.add_argument("--network", default="Conv4_FC3",
                    help="Architecture of the network.")
parser.add_argument("--transfer_learning_autoencoder", default=False, action="store_true",
                    help="If do transfer learning using autoencoder, the learnt weights will be transferred.")
parser.add_argument("--transfer_learning_path", default=None,
                    help='When a path to an experiment is given, will load the autoencoder / CNN weights.')

# Training arguments
parser.add_argument("--num_cnn", default=36, type=int,
                    help="How many CNNs we want to train in a patch-wise way. "
                         "By default, we train each patch from all subjects for one CNN")
parser.add_argument("--epochs", default=1, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Default behaviour will run all splits, else only the splits specified will be run.")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--selection_threshold', default=None, type=float,
                    help='Threshold on the balanced accuracies to compute the subject_level performance '
                         'only based on patches with balanced accuracy > threshold.')

# early stopping arguments
parser.add_argument("--patience", type=int, default=10,
                    help="tolerated epochs without improving for early stopping.")
parser.add_argument("--tolerance", type=float, default=0,
                    help="Tolerance of magnitude of performance after each epoch.")

# Computational issues
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='Uses gpu instead of cpu if cuda is available')


def main(options):

    model = create_model(options.network, options.gpu)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if options.split is None:
        fold_iterator = range(options.n_splits)
    else:
        fold_iterator = [options.split]

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        for i in range(options.num_cnn):

            training_tsv, valid_tsv = load_data(options.diagnosis_tsv_path, options.diagnoses, fi,
                                                n_splits=options.n_splits, baseline=options.baseline)

            print("Running for the %d-th CNN" % i)
            if options.transfer_learning_path is not None:
                if options.transfer_learning_autoencoder:
                    print('Train the model with the weights from a pre-trained autoencoder.')
                    model, _ = load_model_after_ae(model, os.path.join(options.transfer_learning_path, 'best_model_dir',
                                                                       "fold_" + str(fi), 'ConvAutoencoder', 'Encoder'),
                                                   filename='model_best_encoder.pth.tar')
                else:
                    print('Train the model with the weights from a pre-trained CNN.')
                    model, _ = load_model_after_cnn(model,
                                                    os.path.join(options.transfer_learning_path, 'best_model_dir',
                                                                 "fold_" + str(fi), 'CNN', 'best_acc'),
                                                    filename='model_best.pth.tar')
            else:
                print('The model is trained from scratch.')
                model.load_state_dict(init_state)

            data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size,
                                          options.patch_stride, transformations=transformations, patch_index=i,
                                          prepare_dl=options.prepare_dl)
            data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size,
                                          options.patch_stride, transformations=transformations, patch_index=i,
                                          prepare_dl=options.prepare_dl)

            # Use argument load to distinguish training and testing
            train_loader = DataLoader(data_train,
                                      batch_size=options.batch_size,
                                      shuffle=True,
                                      num_workers=options.num_workers,
                                      pin_memory=True
                                      )

            valid_loader = DataLoader(data_valid,
                                      batch_size=options.batch_size,
                                      shuffle=False,
                                      num_workers=options.num_workers,
                                      pin_memory=True
                                      )

            # Define loss and optimizer
            optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                                 options.learning_rate, weight_decay=options.weight_decay)
            loss = torch.nn.CrossEntropyLoss()

            print('Beginning the training task')
            # parameters used in training
            best_accuracy = 0.0
            best_loss_valid = np.inf
            writer_train_batch = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi,
                                                                     "cnn-%i" % i, "train_batch")))
            writer_train_all_data = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi,
                                                                        "cnn-%i" % i, "train_all_data")))
            writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi,
                                                               "cnn-%i" % i, "valid")))

            # initialize the early stopping instance
            early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)

            for epoch in range(options.epochs):
                print("At %i-th epoch." % epoch)

                # train the model
                train_df, acc_mean_train, loss_batch_mean_train, global_step,\
                    = train(model, train_loader, options.gpu, loss, optimizer, writer_train_batch, epoch,
                            model_mode='train')

                # calculate the training accuracy based on all the training data
                train_all_df, acc_mean_train_all, loss_batch_mean_train_all, _,\
                    = train(model, train_loader, options.gpu, loss, optimizer, writer_train_all_data, epoch,
                            model_mode='valid')
                print("For training, subject level balanced accuracy is %f at the end of epoch %d"
                      % (acc_mean_train_all, epoch))

                # at then end of each epoch, we validate one time for the model with the validation data
                valid_df, acc_mean_valid, loss_batch_mean_valid, _\
                    = train(model, valid_loader, options.gpu, loss, optimizer, writer_valid, epoch,
                            model_mode='valid')
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
                    os.path.join(options.output_dir, "best_model_dir", "fold_%i" % fi, "cnn-%i" % i))

                # try early stopping criterion
                if early_stopping.step(loss_batch_mean_valid) or epoch == options.epochs - 1:
                    print("By applying early stopping or at the last epoch defined by user,"
                          "the training is stopped at %d-th epoch" % epoch)

                    break

            for selection in ['best_acc', 'best_loss']:
                # load the best trained model during the training
                model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', 'fold_%i' % fi,
                                                                   'cnn-%i' % i, selection),
                                               gpu=options.gpu, filename='model_best.pth.tar')

                train_df, metrics_train = test(model, train_loader, options.gpu, loss)
                valid_df, metrics_valid = test(model, valid_loader, options.gpu, loss)
                patch_level_to_tsvs(options.output_dir, train_df, metrics_train, fi, selection,
                                    dataset='train', cnn_index=i)
                patch_level_to_tsvs(options.output_dir, valid_df, metrics_valid, fi, selection,
                                    dataset='validation', cnn_index=i)

                torch.cuda.empty_cache()

        for selection in ['best_acc', 'best_loss']:
            soft_voting_to_tsvs(options.output_dir, fi, selection, dataset='train', num_cnn=options.num_cnn,
                                selection_threshold=options.selection_threshold)
            soft_voting_to_tsvs(options.output_dir, fi, selection, dataset='validation', num_cnn=options.num_cnn,
                                selection_threshold=options.selection_threshold)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, "CNN")
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
