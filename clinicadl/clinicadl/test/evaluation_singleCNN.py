# coding: utf8

from __future__ import print_function
import argparse
import os
from os import path
import torch.nn as nn
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning.data import return_dataset, get_transforms, load_data
from clinicadl.tools.deep_learning import read_json
from .test_singleCNN import test_cnn


parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")

# Computational resources
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    # Read json
    model_options = argparse.Namespace()
    json_path = path.join(options.model_path, "commandline_cnn.json")
    model_options = read_json(model_options, json_path=json_path)

    transformations = get_transforms(model_options.mode, model_options.minmaxnormalization)
    criterion = nn.CrossEntropyLoss()

    # Loop on all folds trained
    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    folds_dir = os.listdir(best_model_dir)

    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold %i" % split)

        # Data management
        training_df, valid_df = load_data(model_options.tsv_path, model_options.diagnoses,
                                          split, model_options.n_splits, model_options.baseline)

        data_train = return_dataset(model_options.mode, model_options.input_dir, training_df,
                                    model_options.preprocessing, transformations, model_options)
        data_valid = return_dataset(model_options.mode, model_options.input_dir, valid_df,
                                    model_options.preprocessing, transformations, model_options)

        train_loader = DataLoader(
            data_train,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=True)
        valid_loader = DataLoader(
            data_valid,
            batch_size=options.batch_size,
            shuffle=False,
            num_workers=options.num_workers,
            pin_memory=True)

        test_cnn(options.model_path, train_loader, "train", split, criterion,
                 model_options, options.gpu)
        test_cnn(options.model_path, valid_loader, "validation", split, criterion,
                 model_options, options.gpu)
