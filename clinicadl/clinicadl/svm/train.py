# coding: utf8

import argparse
import os
from .classification_utils import extract_indices_from_5_fold
from .model import VB_KFold_DualSVM
from ..tools.deep_learning import commandline_to_json

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
parser.add_argument("--group_id", type=str, default='ADNIbl',
                    help="Select the group to load the correspondin data in CAPS.")

# Computational issues
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    splits_indices, diagnosis_tsv = extract_indices_from_5_fold(options.diagnosis_path, options.n_splits,
                                                                options.output_dir, baseline=options.baseline,
                                                                diagnoses_list=options.diagnoses)

    wf = VB_KFold_DualSVM(options.input_dir, diagnosis_tsv, options.group_id, "T1", options.output_dir,
                          n_threads=options.num_workers, n_folds=options.n_splits, splits_indices=splits_indices)

    wf.run()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline[0].split = None
    commandline_to_json(commandline, 'SVM')
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
