import argparse
from os import path
import numpy as np
import pickle
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
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument("--n_splits", type=int, default=5,
                    help="Load the data of the corresponding k-fold CV")
parser.add_argument("--split", type=int, default=None,
                    help="Split to perform. Default behaviour will perform all splits.")
parser.add_argument("--group_id", type=str, default='ADNIbl',
                    help="Select the group to load the correspondin data in CAPS.")

# Computational issues
parser.add_argument("--split_indices_path", type=str, default=None,
                    help="If provided, will load fixed split indices for reproducibility issues.")
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


if __name__ == "__main__":
    from .utils import SVMTester, save_data, save_additional_parameters
    from tools.deep_learning.data import load_data
    from tools.deep_learning import commandline_to_json

    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("Unknown arguments: %s" % parser.parse_known_args()[1])
    commandline_to_json(commandline, 'SVM')

    if options.split is None:
        fold_iterator = np.arange(options.n_splits)
    else:
        fold_iterator = [options.split]

    for split in fold_iterator:
        fold_dir = path.join(options.output_dir, 'fold_' + str(split))

        # Load and generate tsv files
        training_df, valid_df = load_data(options.diagnosis_path, options.diagnoses, split,
                                          options.n_splits, options.baseline)

        train_path = save_data(training_df, fold_dir, 'train')
        valid_path = save_data(valid_df, fold_dir, 'validation' + '_' + '_'.join(options.diagnoses))

        # Load split indices
        print("Loading pre-generated indices")
        if options.split_indices_path is not None:
            with open(path.join(options.split_indices_path, 'fold_%i' % split, 'indices.pkl'), 'rb') as file:
                split_indices = pickle.load(file)
        else:
            split_indices = None

        # Run workflow of (Samper-Gonzalez et al., 2018)
        wf = VB_RepHoldOut_DualSVM(options.input_dir,
                                   path.join(train_path, 'sessions.tsv'),
                                   path.join(train_path, 'diagnoses.tsv'),
                                   options.group_id, "T1", fold_dir,
                                   n_threads=options.num_workers, fwhm=8, mask_zeros=True, splits_indices=split_indices)
        wf.run()
        save_additional_parameters(wf, train_path)

        # Test on training and validation sets
        train_dataset = CAPSVoxelBasedInput(options.input_dir,
                                            path.join(train_path, 'sessions.tsv'),
                                            path.join(train_path, 'diagnoses.tsv'),
                                            options.group_id, "T1",
                                            fwhm=8, mask_zeros=False)
        valid_dataset = CAPSVoxelBasedInput(options.input_dir,
                                            path.join(valid_path, 'sessions.tsv'),
                                            path.join(valid_path, 'diagnoses.tsv'),
                                            options.group_id, "T1",
                                            fwhm=8, mask_zeros=False)

        svm_tester = SVMTester(fold_dir)

        valid_path = path.join(fold_dir, 'evaluation', 'validation' + '_' + '_'.join(options.diagnoses))
        svm_tester.test_and_save(valid_dataset, valid_path)

        train_path = path.join(fold_dir, 'evaluation', 'train')
        svm_tester.test_and_save(train_dataset, train_path)
