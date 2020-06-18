# coding: utf8

from __future__ import print_function
import argparse
import os
from os import path
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import test
from ..tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data
from ..tools.deep_learning import create_model, load_model, read_json


def test_cnn(data_loader, subset_name, split, criterion, options):

    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    json_path = path.join(options.model_path, "commandline_cnn.json")
    options = read_json(options, "cnn", json_path=json_path)

    for selection in ["best_acc", "best_loss"]:

        print("Evaluation %s fold %i" % (selection, split))
        model = create_model(options.network)
        fold_dir = "fold_%i" % split
        model_dir = path.join(best_model_dir, fold_dir, 'CNN', selection)
        best_model, best_epoch = load_model(model, model_dir, options.gpu,
                                            filename='model_best.pth.tar')

        metrics_dict, loss, results_df = test(best_model, data_loader, options.gpu, criterion, full_return=True)

        acc = metrics_dict['balanced_accuracy'] * 100
        sen = metrics_dict['sensitivity'] * 100
        spe = metrics_dict['specificity'] * 100
        print("%s, acc %f, loss %f, sensibility %f, specificity %f"
              % (subset_name, acc, loss, sen, spe))

        evaluation_path = path.join(options.model_path, 'performances', fold_dir)
        if not path.exists(path.join(evaluation_path, selection)):
            os.makedirs(path.join(evaluation_path, selection))

        results_df.to_csv(
            path.join(
                evaluation_path,
                selection,
                subset_name + '_image_level_result.tsv'
            ),
            sep='\t', index=False)

        pd.DataFrame(metrics_dict, index=[0]).to_csv(path.join(evaluation_path, selection,
                                                               subset_name + '_image_level_metrics.tsv'),
                                                     sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

    # Mandatory arguments
    parser.add_argument("model_path", type=str,
                        help="Path to the trained model folder.")

    # Model selection
    parser.add_argument("--tsv_path", default=None, type=str)
    parser.add_argument("--selection", default="best_loss", type=str, choices=['best_loss', 'best_acc'],
                        help="Loads the model selected on minimal loss or maximum accuracy on validation.")

    # Computational resources
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="if True computes the visualization on GPU")
    parser.add_argument("--num_workers", '-w', default=8, type=int,
                        help='the number of batch being loaded in parallel')

    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    # Loop on all folds trained
    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    criterion = nn.CrossEntropyLoss()
    folds_dir = os.listdir(best_model_dir)

    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        options.split = split

        # Data management
        training_tsv, valid_tsv = load_data(options.tsv_path, options.diagnoses,
                                            split, options.n_splits, options.baseline)

        if options.minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None

        data_train = MRIDataset(options.caps_dir, training_tsv, options.preprocessing, transform=transformations)
        data_valid = MRIDataset(options.caps_dir, valid_tsv, options.preprocessing, transform=transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  pin_memory=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  pin_memory=True
                                  )

        test_cnn(train_loader, "train", split, criterion, options)
        test_cnn(valid_loader, "validation", split, criterion, options)
