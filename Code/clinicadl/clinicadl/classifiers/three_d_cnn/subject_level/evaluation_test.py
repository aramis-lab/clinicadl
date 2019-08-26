from __future__ import print_function
import argparse
import os
from os import path
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from classifiers.three_d_cnn.subject_level.utils import test
from tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data_test
from tools.deep_learning import load_model, read_json

parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("cohort", type=str,
                    help="Name of the cohort.")

# Data Management
parser.add_argument("--diagnoses", default=None, type=str, nargs='+')
parser.add_argument("--preprocessing", default="linear", choices=["linear", "mniskullstrip", "mni"], type=str,
                    help="Defines the path to data in CAPS.")
parser.add_argument("--selection_eval", default="loss", type=str, choices=['loss', 'accuracy'],
                    help="Loads the model selected on minimal loss or maximum accuracy on validation.")
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization for visualization")
parser.add_argument("--position", default=-1, type=int,
                    help="position of the name in the string given for evaluation.")
parser.add_argument("--split", type=int, default=None, nargs="+",
                    help="Splits on which evaluation is performed. Default behaviour tests all possible folds.")

# Computational ressources
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

    options = read_json(options)

    if "mni" in options.preprocessing:
        options.preprocessing = "mni"
        print(options.preprocessing)

    # Loop on all folds trained
    CNN_dir = os.path.join(options.model_path, 'best_model_dir', 'CNN')
    if options.split is None:
        folds_dir = os.listdir(CNN_dir)
    else:
        folds_dir = [path.join(CNN_dir, 'fold_' + str(fold)) for fold in options.split]

    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold " + str(split))
        model = eval(options.model)()
        if options.gpu:
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()

        if options.selection_eval == 'loss':
            model_dir = os.path.join(CNN_dir, fold_dir, 'best_loss')
            folder_name = 'best_loss'
        else:
            model_dir = os.path.join(CNN_dir, fold_dir, 'best_acc')
            folder_name = 'best_acc'

        best_model, best_epoch = load_model(model, model_dir, options.gpu,
                                            filename='model_best.pth.tar')

        test_tsv = load_data_test(options.diagnosis_path, options.diagnoses)

        if options.minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None

        data_test = MRIDataset(options.input_dir, test_tsv, options.preprocessing, transform=transformations)

        # Use argument load to distinguish training and testing
        test_loader = DataLoader(data_test,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=options.num_workers,
                                 drop_last=False
                                 )

        metrics_test, loss_test, test_df = test(best_model, test_loader, options.gpu, criterion, full_return=True)

        acc_test, sen_test, spe_test = metrics_test['balanced_accuracy'] * 100, metrics_test['sensitivity'] * 100,\
                                       metrics_test['specificity'] * 100
        print("Test, acc %f, loss %f, sensibility %f, specificity %f"
              % (acc_test, loss_test, sen_test, spe_test))

        evaluation_path = path.join(options.model_path, 'performances', fold_dir)
        if not path.exists(path.join(evaluation_path, folder_name)):
            os.makedirs(path.join(evaluation_path, folder_name))

        test_df.to_csv(path.join(evaluation_path, folder_name, 'test-' + options.cohort + '_subject_level_result.tsv'),
                       sep='\t', index=False)

        pd.DataFrame(metrics_test, index=[0]).to_csv(path.join(evaluation_path, folder_name,
                                                               'test-' + options.cohort + '_subject_level_metrics.tsv'),
                                                     sep='\t', index=False)

        del model, best_model

