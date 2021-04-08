# coding: utf8

import argparse
from torch.utils.data import DataLoader
import torch
import os

from clinicadl.tools.deep_learning.iotools import commandline_to_json, return_logger, check_and_clean, \
    write_requirements_version, translate_parameters
from clinicadl.tools.deep_learning.models import load_model, init_model
from clinicadl.tools.deep_learning.data import (load_data,
                                                MRIDatasetSlice,
                                                mix_slices,
                                                generate_sampler,
                                                get_transforms)
from clinicadl.tools.deep_learning.cnn_utils import train, test, mode_level_to_tsvs, get_criterion


def test_bad_cnn(model, output_dir, data_loader, subset_name, split, criterion, mode, logger, gpu=False):

    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(model, os.path.join(output_dir, f'fold-{split}', 'models', selection),
                                       gpu=gpu, filename='model_best.pth.tar')
        results_df, metrics = test(model, data_loader, gpu, criterion, mode)
        logger.info("%s level %s balanced accuracy is %f for model selected on %s"
                    % (mode, subset_name, metrics["balanced_accuracy"], selection))

        mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, mode, dataset=subset_name)


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
    help='Which coordinate axis to take for slicing the MRI. '
         '0 is for saggital, 1 is for coronal and 2 is for axial direction, respectively ')

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
    nargs="+",
    help="Define a specific fold in the k-fold, this is very useful to find the optimal model, "
         "when you do not want to run your k-fold validation")

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

parser.add_argument(
    '--verbose',
    '-v',
    action='count',
    default=0)


def train_bad_cnn(params):
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
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")
    eval_logger = return_logger(params.verbose, "final evaluation")
    check_and_clean(params.output_dir)

    params.mode = "slice"
    params.network_type = "cnn"
    commandline_to_json(params, logger=main_logger)
    write_requirements_version(params.output_dir)
    params = translate_parameters(params)
    train_transforms, all_transforms = get_transforms(params.mode,
                                                      minmaxnormalization=True,
                                                      data_augmentation=False)

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:
        main_logger.info("Fold %i" % fi)

        training_sub_df, valid_sub_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline,
            logger=main_logger,
            multi_cohort=params.multi_cohort
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
                all_transformations=all_transforms,
                train_transformations=train_transforms,
                mri_plane=params.mri_plane,
                prepare_dl=params.prepare_dl,
                mixed=True
                )

        data_valid = MRIDatasetSlice(
                params.caps_directory,
                valid_df,
                all_transformations=all_transforms,
                train_transformations=train_transforms,
                mri_plane=params.mri_plane,
                prepare_dl=params.prepare_dl,
                mixed=True
                )

        train_sampler = generate_sampler(data_train, params.sampler)

        train_loader = DataLoader(
            data_train,
            batch_size=params.batch_size,
            sampler=train_sampler,
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
        model = init_model(params, initial_shape=data_train.size)

        # Define criterion and optimizer
        criterion = get_criterion("default")
        optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                           lr=params.learning_rate,
                                                           weight_decay=params.weight_decay)

        # Define output directories
        log_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(
            params.output_dir, 'fold-%i' % fi, 'models')

        main_logger.debug('Beginning the training task')
        train(model, train_loader, valid_loader, criterion,
              optimizer, False, log_dir, model_dir, params, train_logger)

        test_bad_cnn(model, params.output_dir, train_loader, "train",
                     fi, criterion, params.mode, eval_logger, gpu=params.gpu)
        test_bad_cnn(model, params.output_dir, valid_loader, "validation",
                     fi, criterion, params.mode, eval_logger, gpu=params.gpu)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception(
            "unknown arguments: %s" %
            (parser.parse_known_args()[1]))
    train_bad_cnn(options)
