# coding: utf8

import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning.models import create_model, load_model
from clinicadl.tools.deep_learning.data import (MinMaxNormalization,
                                                load_data,
                                                load_data_test,
                                                MRIDataset_patch)
from clinicadl.tools.deep_learning.cnn_utils import test, sub_level_to_tsvs, soft_voting_to_tsvs

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


def test_cnn(data_loader, subset_name, split, criterion, cnn_index, options):
    for selection in ["best_acc", "best_loss"]:
        # load the best trained model during the training
        model = create_model(options.network, options.gpu)
        model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', "fold_%i" % split,
                                                           'cnn-%i' % cnn_index, selection),
                                       gpu=options.gpu, filename='model_best.pth.tar')

        results_df, metrics = test(model, data_loader, options.gpu, criterion, options.mode)
        print("Patch level balanced accuracy is %f" % metrics['balanced_accuracy'])

        sub_level_to_tsvs(options.output_dir, results_df, metrics, split, selection, options.mode,
                          dataset=subset_name, cnn_index=cnn_index)


parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch-level multi-CNN for test the trained classifiers")

# Mandatory argument
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument('preprocessing', type=str,
                    help='Defines the type of preprocessing of CAPS data.',
                    choices=['t1-linear', 't1-extensive'])
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to the tsv containing all the test dataset")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs and the tsv files containing the performances.")

# Data management
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--stride_size", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument('--dataset', default="validation",
                    help="If the evaluation on the validation set is wanted, must be set to 'validation'. "
                         "Otherwise must be named with the form 'test-cohort_name'.")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Default behaviour will run all splits, else only the splits specified will be run.")
parser.add_argument('--prepare_dl', default=False, action="store_true",
                    help="If True the outputs of preprocessing prepare_dl are used, else the whole MRI is loaded.")

# test arguments
parser.add_argument("--network", default="Conv4_FC3",
                    help="Architecture of the network.")
parser.add_argument("--num_cnn", default=36, type=int,
                    help="How many CNNs we want to train in a patch-wise way."
                         "By default, we train each patch from all subjects for one CNN.")
parser.add_argument("--diagnoses", default=["sMCI", "pMCI"], type=str, nargs="+",
                    help="Labels based on binary classification.")
parser.add_argument('--selection_threshold', default=None, type=float,
                    help='Threshold on the balanced accuracies to compute the subject-level performance '
                         'only based on patches with balanced accuracy > threshold.')

# Computational issues
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--gpu", default=False, action='store_true',
                    help="If use gpu or cpu. Empty implies cpu usage.")


def main(options):

    transformations = transforms.Compose([MinMaxNormalization()])
    criterion = torch.nn.CrossEntropyLoss()

    if options.split is None:
        fold_iterator = range(options.n_splits)
    else:
        fold_iterator = options.split

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        if options.dataset == 'validation':
            _, test_df = load_data(options.diagnosis_tsv_path, options.diagnoses, fi,
                                   n_splits=options.n_splits, baseline=True)
        else:
            test_df = load_data_test(options.diagnosis_tsv_path, options.diagnoses)

        for n in range(options.num_cnn):

            dataset = MRIDataset_patch(
                    options.caps_directory,
                    test_df,
                    options.patch_size,
                    options.stride_size,
                    preprocessing=options.preprocessing,
                    transformations=transformations,
                    patch_index=n,
                    prepare_dl=options.prepare_dl)

            test_loader = DataLoader(
                    dataset,
                    batch_size=options.batch_size,
                    shuffle=False,
                    num_workers=options.num_workers,
                    pin_memory=True)

            test_cnn(test_loader, options.dataset, fi, criterion, n, options)

        for selection in ['best_acc', 'best_loss']:
            soft_voting_to_tsvs(
                options.output_dir,
                fi,
                selection,
                mode="patch",
                dataset=options.dataset,
                num_cnn=options.num_cnn,
                selection_threshold=options.selection_threshold
            )


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
