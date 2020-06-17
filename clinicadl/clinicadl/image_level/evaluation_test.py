# coding: utf8

from __future__ import print_function
import argparse
import os
from os import path
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import test
from clinicadl.tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data_test
from clinicadl.tools.deep_learning import create_model, load_model, read_json

parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("caps_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("tsv_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("cohort", type=str,
                    help="Name of the cohort.")

# Data Management
parser.add_argument("--diagnoses", default=None, type=str, nargs='+',
                    help='Default will load the same diagnoses used in training.')
parser.add_argument("--selection", default="best_loss", type=str, choices=['best_loss', 'best_acc'],
                    help="Loads the model selected on minimal loss or maximum accuracy on validation.")

# Computational resources
parser.add_argument("--batch_size", default=16, type=int,
                    help='Size of the batch loaded by the data loader.')
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    # Loop on all folds trained
    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    folds_dir = os.listdir(best_model_dir)

    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold %i" % split)
        model_options = argparse.Namespace()
        json_path = path.join(options.model_path, "commandline_CNN.json")
        model_options = read_json(model_options, "CNN", json_path=json_path)
        model = create_model(model_options.network, options.gpu)

        criterion = nn.CrossEntropyLoss()

        model_dir = os.path.join(best_model_dir, fold_dir, 'CNN', options.selection)
        best_model, best_epoch = load_model(model, model_dir, options.gpu,
                                            filename='model_best.pth.tar')

        # Load test data
        if options.diagnoses is None:
            options.diagnoses = model_options.diagnoses

        test_tsv = load_data_test(options.tsv_path, options.diagnoses)

        if model_options.minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None

        data_test = MRIDataset(
                options.caps_dir,
                test_tsv,
                model_options.preprocessing,
                transform=transformations
                )

        test_loader = DataLoader(
                data_test,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                pin_memory=True
                )

        # Run test
        metrics_test, loss_test, test_df = test(
                best_model,
                test_loader,
                options.gpu,
                criterion,
                full_return=True)

        acc_test = metrics_test['balanced_accuracy'] * 100
        sen_test = metrics_test['sensitivity'] * 100
        spe_test = metrics_test['specificity'] * 100
        print("Test, acc %f, loss %f, sensibility %f, specificity %f"
              % (acc_test, loss_test, sen_test, spe_test))

        evaluation_path = path.join(options.model_path, 'performances', fold_dir)
        if not path.exists(path.join(evaluation_path, options.selection)):
            os.makedirs(path.join(evaluation_path, options.selection))

        test_df.to_csv(path.join(evaluation_path, options.selection,
                                 'test-' + options.cohort + '_image_level_result.tsv'), sep='\t', index=False)

        pd.DataFrame(metrics_test, index=[0]).to_csv(path.join(evaluation_path, options.selection,
                                                               'test-' + options.cohort + '_image_level_metrics.tsv'),
                                                     sep='\t', index=False)

        del model, best_model
