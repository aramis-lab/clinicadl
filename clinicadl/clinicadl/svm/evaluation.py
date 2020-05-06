# coding: utf8

import argparse
import os
from os import path
import pandas as pd
import numpy as np
from .classification_utils import load_data_svm, revert_mask, CAPSVoxelBasedInput, evaluate_prediction, save_data
from ..tools.deep_learning.data import load_data, load_data_test

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("caps_directory_source", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("caps_directory_target", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("output_dir", type=str,
                    help="Path to the output directory.")

# Data Management
parser.add_argument("--diagnoses", "-d", default=['AD', 'CN'], nargs='+', type=str,
                    help="The diagnoses used for the classification")
parser.add_argument("--set", type=str, default="validation",
                    choices=["test-ADNI", "test-AIBL", "test-OASIS", "validation"],
                    help="Set to use for evaluation")
parser.add_argument("--n_splits", type=int, default=5,
                    help="Load the data of the corresponding k-fold CV")
parser.add_argument("--group_id_source", type=str, default='ADNIbl',
                    help="Select the group to load the corresponding data in CAPS.")
parser.add_argument("--group_id_target", type=str, default='ADNIbl',
                    help="Select the group to load the corresponding data in CAPS.")

# Computational issues
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):
    # reconstruct the trained model from the source task
    diagnosis_tsv_all = os.path.join(options.output_dir, 'all_subjects.tsv')
    source_data = CAPSVoxelBasedInput(options.caps_directory_source, diagnosis_tsv_all, options.group_id_source,
                                      "T1", fwhm=8, mask_zeros=False)
    _, source_orig_shape, source_mask = load_data_svm(source_data.get_images(), mask=True)

    folder_name = options.set + '_' + '_'.join(options.diagnoses)
    result_dir = path.join(options.output_dir, folder_name)
    if not path.exists(result_dir):
        os.makedirs(result_dir)

    res_final = pd.DataFrame()
    for fi in range(options.n_splits):

        # Load and generate tsv files
        if options.set == "validation":
            _, test_df = load_data(options.diagnosis_path, options.diagnoses, fi, options.n_splits, True)
        else:
            test_df = load_data_test(options.diagnosis_path, options.diagnoses)

        test_path = save_data(test_df, options.output_dir, path.join('fold_%i' % fi, folder_name))

        target_data = CAPSVoxelBasedInput(options.caps_directory_target, test_path,
                                          options.group_id_target, "T1", fwhm=8, mask_zeros=False)

        subjects = list(test_df.participant_id)
        sessions = list(test_df.session_id)
        subjects_sessions = [subjects[i] + '_' + sessions[i] for i in range(len(subjects))]

        weights = np.loadtxt(path.join(options.output_dir, 'classifier', 'fold_' + str(fi), 'weights.txt'))

        w = revert_mask(weights, source_mask, source_orig_shape).flatten()
        b = np.loadtxt(path.join(options.output_dir, 'classifier', 'fold_' + str(fi), 'intersect.txt'))

        target_image = target_data.get_x()
        target_label = target_data.get_y()

        y_hat = np.dot(w, target_image.transpose()) + b
        y_binary = (y_hat < 0) * 1

        evaluation = evaluate_prediction(list(target_label), list(y_binary))

        del evaluation['confusion_matrix']
        res_df = pd.DataFrame(evaluation, index=[fi])

        subjects_df = pd.DataFrame({'subject': np.array(subjects_sessions),
                                    'y': target_label, 'y_hat': y_binary, 'fold_index': fi})
        subjects_df.to_csv(path.join(result_dir, 'subjects_fold-' + str(fi) + '.tsv'),
                           index=False, sep='\t', encoding='utf-8')
        res_final = res_final.append(res_df)
    res_final.to_csv(path.join(result_dir, 'results.tsv'), sep='\t', index=False)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
