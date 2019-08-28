import argparse
import os
from os import path
import pandas as pd
from clinica.pipelines.machine_learning.input import CAPSVoxelBasedInput
from clinica.pipelines.machine_learning.ml_workflows import VB_RepHoldOut_DualSVM

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("output_dir", type=str,
                    help="Path to the outputs directory.")

# Data Management
parser.add_argument("--diagnoses", "-d", default=['AD', 'CN'], nargs='+', type=str,
                    help="The diagnoses used for the classification")
parser.add_argument("--set", type=str, default="validation",
                    choices=["test-ADNI", "test-AIBL", "test-OASIS", "validation"],
                    help="Set to use for evaluation")
parser.add_argument("--n_splits", type=int, default=5,
                    help="Load the data of the corresponding k-fold CV")
parser.add_argument("--group_id", type=str, default='ADNIbl',
                    help="Select the group to load the corresponding data in CAPS.")

# Computational issues
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


if __name__ == "__main__":
    from .utils import SVMTester, save_data
    from tools.deep_learning.data import load_data, load_data_test

    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("Unknown arguments: %s" % parser.parse_known_args()[1])

    folder_name = options.set + '_' + '_'.join(options.diagnoses)

    for split in range(options.n_splits):
        fold_dir = path.join(options.output_dir, 'fold_' + str(split))

        # Load and generate tsv files
        if options.set == "validation":
            _, test_df = load_data(options.diagnosis_path, options.diagnoses, split, options.n_splits, True)
        else:
            test_df = load_data_test(options.diagnosis_path, options.diagnoses)

        test_path = save_data(test_df, fold_dir, folder_name)

        # Test on test set
        test_dataset = CAPSVoxelBasedInput(options.input_dir,
                                           path.join(test_path, 'sessions.tsv'),
                                           path.join(test_path, 'diagnoses.tsv'),
                                           options.group_id, "T1",
                                           fwhm=8, mask_zeros=False)

        svm_tester = SVMTester(fold_dir)

        test_path = path.join(fold_dir, 'evaluation', folder_name)
        svm_tester.test_and_save(test_dataset, test_path)
