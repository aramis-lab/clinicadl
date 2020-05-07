# coding: utf8

import argparse
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import copy
import torch
import os
import numpy as np
from time import time

from utils import mix_slices, train, test, slice_level_to_tsvs, soft_voting_to_tsvs
from clinicadl.tools.deep_learning import (EarlyStopping,
                                           create_model,
                                           save_checkpoint,
                                           load_model,
                                           commandline_to_json)
from clinicadl.tools.deep_learning.data import (load_data,
                                                MinMaxNormalization,
                                                MRIDataset_slice_mixed)


__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018-2020 The Aramis Lab Team"
__credits__ = ["Junhao Wen" "Elina Thibeau-Sutre" "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"

# Train 2D CNN - Slice level network
# The input MRI's dimension is 169*208*179 after cropping"
parser = argparse.ArgumentParser(
        description="Argparser for Pytorch 2D CNN, The input MRI's dimension is 169*208*179 after cropping")

# Mandatory argument
parser.add_argument(
        "caps_directory",
        type=str,
        help="Path to the caps of image processing pipeline of DL")

parser.add_argument(
        "tsv_path",
        type=str,
        help="Path to tsv file of the population based on the diagnosis tsv files. "
             "To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument(
        "output_dir",
        type=str,
        help="Path to store the classification outputs, and the tsv files containing the performances.")

# Data argument
parser.add_argument(
        "--mri_plane",
        default=0,
        type=int,
        help='Which coordinate axis to take for slicing the MRI. 0 is for saggital, 1 is for coronal and 2 is for axial direction, respectively ')

parser.add_argument(
        '--baseline',
        default=False,
        action="store_true",
        help="Using baseline scans or all available longitudinal scans for training")

# train argument
parser.add_argument(
        "--network",
        default="resnet18",
        help="Deep network type. Only ResNet was designed for training from scratch.")

parser.add_argument(
        "--diagnoses",
        default=["AD", "CN"],
        type=str,
        nargs="+",
        help="Labels for any binary task")

parser.add_argument(
        "--learning_rate", "-lr",
        default=1e-3,
        type=float,
        help="Learning rate of the optimization. (default=0.01)")

parser.add_argument(
        "--n_splits",
        default=5,
        type=int,
        help="Define the cross validation, by default, we use 5-fold.")

parser.add_argument(
        "--split",
        default=None,
        type=int,
        help="Define a specific fold in the k-fold, this is very useful to find the optimal model, where you do not want to run your k-fold validation")

parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Epochs through the data. (default=20)")

parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training. (default=1)")

parser.add_argument(
        "--optimizer",
        default="Adam",
        choices=["SGD", "Adadelta", "Adam"],
        help="Optimizer of choice for training. (default=Adam)")

parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="If use gpu or cpu. Empty implies cpu usage.")

parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help='the number of batch being loaded in parallel')

parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help='weight decay (default: 1e-4)')

parser.add_argument(
        '--selection_threshold',
        default=None,
        type=float,
        help='Threshold on the balanced accuracies to compute the subject_level '
             'performance only based on patches with balanced accuracy > threshold.')

parser.add_argument(
        '--prepare_dl',
        default=False,
        action="store_true",
        help="If True the outputs of preprocessing prepare_dl are used, else the whole MRI is loaded.")

# early stopping arguments
parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="tolerated epochs without improving for early stopping.")

parser.add_argument(
        "--tolerance",
        type=float,
        default=0,
        help="Tolerance of magnitude of performance after each epoch.")


def train_CNN_bad_data_split(params):

    # Initialize the model
    print('Do transfer learning with existed model trained on ImageNet!\n')
    print('The chosen network is %s !' % params.network)

    model = create_model(params.network, params.gpu)
    trg_size = (224, 224)  # most of the imagenet pretrained model has this input size

    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H
    # and W are expected to be at least 224. The images have to be loaded in to
    # a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
    # and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])

    total_time = time()
    init_state = copy.deepcopy(model.state_dict())

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]

    for fi in fold_iterator:
        print("Running for the %d-th fold" % fi)

        training_sub_df, valid_sub_df = load_data(
                params.tsv_path,
                params.diagnoses,
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline
                )

        # split the training + validation by slice
        training_df, valid_df = mix_slices(
                training_sub_df,
                valid_sub_df,
                mri_plane=params.mri_plane
                )

        data_train = MRIDataset_slice_mixed(
                params.caps_directory,
                training_df,
                transformations=transformations,
                mri_plane=params.mri_plane,
                prepare_dl=params.prepare_dl
                )

        data_valid = MRIDataset_slice_mixed(
                params.caps_directory,
                valid_df,
                transformations=transformations,
                mri_plane=params.mri_plane,
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

        # chosen optimizer for back-propagation
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),  params.learning_rate, weight_decay=params.weight_decay)

        model.load_state_dict(init_state)

        # Binary cross-entropy loss
        loss = torch.nn.CrossEntropyLoss()

        # parameters used in training
        best_accuracy = 0.0
        best_loss_valid = np.inf

        writer_train_batch = SummaryWriter(
            log_dir=(os.path.join(
                params.output_dir,
                "log_dir",
                "fold_" + str(fi),
                "train_batch")
                )
            )
        writer_train_all_data = SummaryWriter(
            log_dir=(os.path.join(
                params.output_dir,
                "log_dir",
                "fold_" + str(fi),
                "train_all_data")
                )
            )

        writer_valid = SummaryWriter(
                log_dir=(os.path.join(
                    params.output_dir,
                    "log_dir",
                    "fold_" + str(fi),
                    "valid")
                    )
                )

        # initialize the early stopping instance
        early_stopping = EarlyStopping(
                'min',
                min_delta=params.tolerance,
                patience=params.patience
                )

        for epoch in range(params.epochs):
            print("At %i-th epoch." % epoch)

            # train the model
            train_df, acc_mean_train, loss_batch_mean_train, global_step \
                = train(
                        model,
                        train_loader,
                        params,
                        loss,
                        optimizer,
                        writer_train_batch,
                        epoch,
                        model_mode='train',
                        selection_threshold=params.selection_threshold
                        )

            # calculate the accuracy with the whole training data to monitor overfitting
            train_all_df, acc_mean_train_all, loss_batch_mean_train_all, _\
                = train(
                        model,
                        train_loader,
                        params.gpu,
                        loss,
                        optimizer,
                        writer_train_all_data,
                        epoch,
                        model_mode='valid',
                        selection_threshold=params.selection_threshold
                        )

            print("For training, subject level balanced accuracy is %f at the end of epoch %d" % (acc_mean_train_all, epoch))

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
                    os.path.join(params.output_dir,
                                 'best_model_dir',
                                 'fold_%i' % fi,
                                 'CNN',
                                 str(selection)),
                    gpu=params.gpu,
                    filename='model_best.pth.tar'
                    )

            train_df, metrics_train = test(model, train_loader, params.gpu, loss)
            valid_df, metrics_valid = test(model, valid_loader, params.gpu, loss)

            # write the information of subjects and performances into tsv files.
            slice_level_to_tsvs(params.output_dir, train_df, metrics_train, fi,
                                dataset='train', selection=selection)
            slice_level_to_tsvs(params.output_dir, valid_df, metrics_valid, fi,
                                dataset='validation', selection=selection)

            soft_voting_to_tsvs(params.output_dir, fi, dataset='train', selection=selection,
                                selection_threshold=params.selection_threshold)
            soft_voting_to_tsvs(params.output_dir, fi, dataset='validation', selection=selection,
                                selection_threshold=params.selection_threshold)
            torch.cuda.empty_cache()

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, "CNN")
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    train_CNN_bad_data_split(options)
