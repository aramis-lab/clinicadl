# coding: utf8

"""
Check the absence of data leakage
    1) Baseline datasets contain only one scan per subject
    2) No intersection between train and test sets
    3) Absence of MCI train subjects in test sets of subcategories of MCI
"""

import argparse
import pandas as pd
import os
from os import path


def check_subject_unicity(diagnosis_path):
    print('Check unicity', diagnosis_path)
    diagnosis_df_paths = os.listdir(diagnosis_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('_baseline.tsv')]

    for diagnosis_df_path in diagnosis_df_paths:
        flag_unique = True
        check_df = pd.read_csv(path.join(diagnosis_path, diagnosis_df_path), sep='\t')
        check_df.set_index(['participant_id', 'session_id'], inplace=True)
        for subject, subject_df in check_df.groupby(level=0):
            if len(subject_df) > 1:
                flag_unique = False

        assert flag_unique


def check_independance(train_path, test_path):
    print('Check independence')
    diagnosis_df_paths = os.listdir(train_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('_baseline.tsv')]

    for diagnosis_df_path in diagnosis_df_paths:
        flag_independant = True
        train_df = pd.read_csv(path.join(train_path, diagnosis_df_path), sep='\t')
        train_df.set_index(['participant_id', 'session_id'], inplace=True)
        test_df = pd.read_csv(path.join(test_path, diagnosis_df_path), sep='\t')
        test_df.set_index(['participant_id', 'session_id'], inplace=True)

        for subject, session in train_df.index:
            if subject in test_df.index:
                flag_independant = False

        assert flag_independant


def check_subgroup_independence(train_path, test_path):
    print('Check subgroup independence')
    diagnosis_df_paths = os.listdir(test_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('_baseline.tsv')]
    sub_diagnosis_list = [x for x in diagnosis_df_paths if 'MCI' in x and x != 'MCI_baseline.tsv']

    MCI_train_df = pd.read_csv(path.join(train_path, 'MCI_baseline.tsv'), sep='\t')
    MCI_train_df.set_index(['participant_id', 'session_id'], inplace=True)
    for sub_diagnosis in sub_diagnosis_list:
        flag_independant = True
        sub_test_df = pd.read_csv(path.join(test_path, sub_diagnosis), sep='\t')
        sub_test_df.set_index(['participant_id', 'session_id'], inplace=True)

        for subject, session in MCI_train_df.index:
            if subject in sub_test_df.index:
                flag_independant = False

        assert flag_independant

    MCI_test_df = pd.read_csv(path.join(test_path, 'MCI_baseline.tsv'), sep='\t')
    MCI_test_df.set_index(['participant_id', 'session_id'], inplace=True)
    for sub_diagnosis in sub_diagnosis_list:
        flag_independant = True
        sub_test_df = pd.read_csv(path.join(train_path, sub_diagnosis), sep='\t')
        sub_test_df.set_index(['participant_id', 'session_id'], inplace=True)

        for subject, session in MCI_test_df.index:
            if subject in sub_test_df.index:
                flag_independant = False

        assert flag_independant


def run_test_suite(formatted_data_path, n_splits, subset_name):
    check_train = True

    if n_splits == 0:
        train_path = path.join(formatted_data_path, 'train')
        test_path = path.join(formatted_data_path, subset_name)
        if not path.exists(train_path):
            check_train = False

        check_subject_unicity(test_path)
        if check_train:
            check_subject_unicity(train_path)
            check_independance(train_path, test_path)
            MCI_path = path.join(train_path, 'MCI_baseline.tsv')
            if path.exists(MCI_path):
                check_subgroup_independence(train_path, test_path)

    else:
        for split in range(n_splits):
            train_path = path.join(formatted_data_path, 'train_splits-' + str(n_splits),
                                   'split-' + str(split))
            test_path = path.join(formatted_data_path, subset_name + '_splits-' + str(n_splits),
                                  'split-' + str(split))

            if not path.exists(train_path):
                check_train = False

            check_subject_unicity(test_path)
            if check_train:
                check_subject_unicity(train_path)
                check_independance(train_path, test_path)
                MCI_path = path.join(train_path, 'MCI_baseline.tsv')
                if path.exists(MCI_path):
                    check_subgroup_independence(train_path, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparser test")

    parser.add_argument("formatted_data_path", type=str,
                        help="Path to the folder containing formatted data.")
    parser.add_argument("--n_splits", "-n", type=int, default=0,
                        help="Number of splits. If 0 the unique train / test split is studied")
    parser.add_argument("--subset_name", type=str, default="test",
                        help="Name of the subset that is complementary to train.")

    args = parser.parse_args()

    run_test_suite(args.formatted_data_path, args.n_splits, args.subset_name)
