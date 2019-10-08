from os import path
import os
import argparse
import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to the outputs directory.")

# Data Management
parser.add_argument("--diagnoses", "-d", default=['AD', 'CN'], nargs='+', type=str,
                    help="The diagnoses used for the classification")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument("--n_splits", type=int, default=5,
                    help="Load the data of the corresponding k-fold CV")

# Split arguments
parser.add_argument("--n_iterations", type=int, default=100,
                    help="Number of iterations performed for the repeated hold out.")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of the training that is used as test during the repeated hold out.")

if __name__ == "__main__":
    from .classification_utils import diagnosis_code
    from ..tools.deep_learning.data import load_data

    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("Unknown arguments: %s" % parser.parse_known_args()[1])

    for split in range(options.n_splits):

        train_df, _ = load_data(options.diagnosis_path, options.diagnoses, split, options.n_splits, options.baseline)

        for diagnosis in diagnosis_code.keys():
            train_df.loc[train_df[train_df.diagnosis == diagnosis].index, 'diagnosis_code'] = diagnosis_code[diagnosis]

        y = np.array(train_df.diagnosis_code)

        splits = StratifiedShuffleSplit(n_splits=options.n_iterations, test_size=options.test_size)
        splits_indices = list(splits.split(np.zeros(len(y)), y))

        if not path.exists(path.join(options.output_dir, 'fold_%i' % split)):
            os.makedirs(path.join(options.output_dir, 'fold_%i' % split))

        with open(path.join(options.output_dir, 'fold_%i' % split, 'indices.pkl'), 'wb') as s:
            pickle.dump(splits_indices, s)
