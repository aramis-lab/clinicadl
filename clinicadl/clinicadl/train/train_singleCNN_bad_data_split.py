# coding: utf8

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os
from time import time

from clinicadl.tools.deep_learning import commandline_to_json
from clinicadl.tools.deep_learning.models import load_model, init_model
from clinicadl.tools.deep_learning.data import (load_data,
                                                MinMaxNormalization,
                                                MRIDatasetSlice,
                                                mix_slices)
from clinicadl.tools.deep_learning.cnn_utils import train, test, mode_level_to_tsvs


__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018-2020 The Aramis Lab Team"
__credits__ = ["Junhao Wen" "Elina Thibeau-Sutre" "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"


def test_cnn(data_loader, subset_name, split, criterion, options):

    for selection in ["best_acc", "best_loss"]:
        # load the best trained model during the training
        model = init_model(
            options.model,
            gpu=options.gpu,
            dropout=options.dropout)
        model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', "fold_%i" % split,
                                                           'CNN', selection),
                                       gpu=options.gpu, filename='model_best.pth.tar')

        results_df, metrics = test(
            model, data_loader, options.gpu, criterion, options.mode)
        print(
            "Slice level balanced accuracy is %f" %
            metrics['balanced_accuracy'])

        mode_level_to_tsvs(
            options.output_dir,
            results_df,
            metrics,
            split,
            selection,
            options.mode,
            dataset=subset_name)


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
    "--model",
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
    "--dropout",
    default=0,
    type=float,
    help="Rate of dropout applied to dropout layers."
)

parser.add_argument(
    '--selection_threshold',
    default=None,
    type=float,
    help='Threshold on the balanced accuracies to compute the subject-level '
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
    print('The chosen network is %s !' % params.model)

    # most of the imagenet pretrained model has this input size
    trg_size = (224, 224)

    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H
    # and W are expected to be at least 224. The images have to be loaded in to
    # a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
    # and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])
    params.dropout = 0.8

    total_time = time()

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

        data_train = MRIDatasetSlice(
                params.caps_directory,
                training_df,
                transformations=transformations,
                mri_plane=params.mri_plane,
                prepare_dl=params.prepare_dl,
                mixed=True
                )

        data_valid = MRIDatasetSlice(
                params.caps_directory,
                valid_df,
                transformations=transformations,
                mri_plane=params.mri_plane,
                prepare_dl=params.prepare_dl,
                mixed=True
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

        # Initialize the model
        print('Initialization of the model')
        model = init_model(
            params.model,
            gpu=params.gpu,
            dropout=params.dropout)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' %
            fi, 'tensorboard_logs')
        model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models')

        print('Beginning the training task')
        train(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            False,
            log_dir,
            model_dir,
            params)

        test_cnn(train_loader, "train", fi, criterion, options)
        test_cnn(valid_loader, "validation", fi, criterion, options)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, "CNN")
    options = commandline[0]
    if commandline[1]:
        raise Exception(
            "unknown arguments: %s" %
            (parser.parse_known_args()[1]))
    train_CNN_bad_data_split(options)
