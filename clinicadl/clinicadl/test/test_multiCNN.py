# coding: utf8

import argparse
import os
from os import path
import torch
from torch.utils.data import DataLoader

from clinicadl.tools.deep_learning import read_json
from clinicadl.tools.deep_learning.models import create_model, load_model
from clinicadl.tools.deep_learning.data import (get_transforms,
                                                load_data_test,
                                                return_dataset,
                                                compute_num_cnn)
from clinicadl.tools.deep_learning.cnn_utils import test, mode_level_to_tsvs, soft_voting_to_tsvs


def test_cnn(output_dir, data_loader, subset_name, split, criterion, cnn_index, model_options, gpu=False):
    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model = create_model(model_options.model, gpu, dropout=model_options.dropout)
        model, best_epoch = load_model(model, os.path.join(output_dir, 'fold-%i' % split, 'models',
                                                           'cnn-%i' % cnn_index, selection),
                                       gpu=gpu, filename='model_best.pth.tar')

        results_df, metrics = test(model, data_loader, gpu, criterion, model_options.mode)
        print("%s level balanced accuracy is %f" % (model_options.mode, metrics['balanced_accuracy']))

        mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, model_options.mode,
                           dataset=subset_name, cnn_index=cnn_index)


parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch-level multi-CNN for test the trained classifiers")

# Mandatory argument
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("tsv_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("dataset", type=str,
                    help="Name of the dataset on which the classification is performed.")

# Data Management
parser.add_argument("--diagnoses", default=None, type=str, nargs='+',
                    help='Default will load the same diagnoses used in training.')

# Computational resources
parser.add_argument("--batch_size", default=16, type=int,
                    help='Size of the batch loaded by the data loader.')
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")


def main(options):

    # Read json
    model_options = argparse.Namespace()
    json_path = path.join(options.model_path, "commandline_cnn.json")
    model_options = read_json(model_options, json_path=json_path)
    num_cnn = compute_num_cnn(options.input_dir, options.tsv_path, model_options, data="test")

    # Load test data
    if options.diagnoses is None:
        options.diagnoses = model_options.diagnoses

    test_df = load_data_test(options.tsv_path, options.diagnoses)
    transformations = get_transforms(model_options.mode, model_options.minmaxnormalization)
    criterion = torch.nn.CrossEntropyLoss()

    # Loop on all folds trained
    best_model_dir = os.path.join(options.model_path, 'best_model_dir')
    folds_dir = os.listdir(best_model_dir)

    # Loop on folds
    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold %i" % split)

        for cnn_index in range(num_cnn):
            dataset = return_dataset(model_options.mode, options.input_dir, test_df, options.preprocessing,
                                     transformations, options, cnn_index=cnn_index)

            test_loader = DataLoader(
                dataset,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                pin_memory=True)

            test_cnn(options.model_path, test_loader, options.dataset, split, criterion, cnn_index,
                     model_options, options.gpu)

        for selection in ['best_acc', 'best_loss']:
            soft_voting_to_tsvs(
                options.model_path,
                split,
                selection,
                mode=options.mode,
                dataset=options.dataset,
                num_cnn=num_cnn,
                selection_threshold=model_options.selection_threshold
            )


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
