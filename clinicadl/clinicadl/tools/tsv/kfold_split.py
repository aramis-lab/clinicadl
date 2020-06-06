# coding: utf8

from .tsv_utils import baseline_df
import shutil
from sklearn.model_selection import StratifiedKFold
from os import path
import os
import pandas as pd
import numpy as np

sex_dict = {'M': 0, 'F': 1}


def split_diagnoses(formatted_data_path,
                    n_splits=5, subset_name="validation", MCI_sub_categories=True):
    """
    Performs a k-fold split for each label independently on the subject level.
    The train folder will contain two lists per fold per diagnosis (baseline and longitudinal),
    whereas the test folder will only include the list of baseline sessions for each fold.

    Args:
        formatted_data_path (str): Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        n_splits (int): Number of folds in the k-fold split.
        subset_name (str): Name of the subset that is complementary to train.
        MCI_sub_categories (bool): If True, manages MCI sub-categories to avoid data leakage.

    Returns:
         writes three files per split per <label>.tsv file present in formatted_data_path:
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>.tsv
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>_baseline.tsv
            - formatted_data_path/<subset_name>_splits-<n_splits>/split-<split>/<label>_baseline.tsv
    """

    # Read files
    results_path = formatted_data_path

    train_path = path.join(results_path, 'train_splits-' + str(n_splits))
    if path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path)
    for i in range(n_splits):
        os.mkdir(path.join(train_path, 'split-' + str(i)))

    test_path = path.join(results_path, subset_name + '_splits-' + str(n_splits))
    if path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)
    for i in range(n_splits):
        os.mkdir(path.join(test_path, 'split-' + str(i)))

    diagnosis_df_paths = os.listdir(results_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.tsv')]
    diagnosis_df_paths = [x for x in diagnosis_df_paths if not x.endswith('_baseline.tsv')]

    MCI_special_treatment = False

    if MCI_sub_categories and 'MCI.tsv' in diagnosis_df_paths:
        diagnosis_df_paths.remove('MCI.tsv')
        MCI_special_treatment = True

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis_df_path in diagnosis_df_paths:
        print(diagnosis_df_path)
        diagnosis = diagnosis_df_path.split('.')[0]
        print(diagnosis)

        diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path), sep='\t')
        diagnosis_baseline_df = baseline_df(diagnosis_df, diagnosis)
        diagnoses_list = list(diagnosis_baseline_df.diagnosis)
        unique = list(set(diagnoses_list))
        y = np.array(
            [unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):

            train_index, test_index = indices

            test_df = diagnosis_baseline_df.iloc[test_index]
            train_df = diagnosis_baseline_df.iloc[train_index]
            # Retrieve all sessions for the training set
            complete_train_df = pd.DataFrame()
            for idx in train_df.index.values:
                subject = train_df.loc[idx, 'participant_id']
                subject_df = diagnosis_df[diagnosis_df.participant_id == subject]
                complete_train_df = pd.concat([complete_train_df, subject_df])

            complete_train_df.to_csv(path.join(train_path, 'split-' + str(i), str(diagnosis) + '.tsv'),
                                     sep='\t', index=False)
            train_df.to_csv(
                path.join(train_path, 'split-' + str(i), str(diagnosis) + '_baseline.tsv'), sep='\t', index=False)
            test_df.to_csv(
                path.join(test_path, 'split-' + str(i), str(diagnosis) + '_baseline.tsv'), sep='\t', index=False)

    if MCI_special_treatment:

        # Extraction of MCI subjects without intersection with the sMCI / pMCI train
        diagnosis_df = pd.read_csv(path.join(results_path, 'MCI.tsv'), sep='\t')
        MCI_df = diagnosis_df.set_index(['participant_id', 'session_id'])
        supplementary_diagnoses = []

        print('Before subjects removal')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        print('%i subjects, %i scans' % (len(sub_df), len(diagnosis_df)))

        if 'sMCI.tsv' in diagnosis_df_paths:
            sMCI_baseline_df = pd.read_csv(path.join(results_path, 'sMCI_baseline.tsv'), sep='\t')
            for idx in sMCI_baseline_df.index.values:
                subject = sMCI_baseline_df.loc[idx, 'participant_id']
                MCI_df.drop(subject, inplace=True, level=0)
            supplementary_diagnoses.append('sMCI')

            print('Removed %i subjects' % len(sMCI_baseline_df))
            sub_df = MCI_df.reset_index().groupby('participant_id')['session_id'].nunique()
            print('%i subjects, %i scans' % (len(sub_df), len(MCI_df)))

        if 'pMCI.tsv' in diagnosis_df_paths:
            pMCI_baseline_df = pd.read_csv(path.join(results_path, 'pMCI_baseline.tsv'), sep='\t')
            for idx in pMCI_baseline_df.index.values:
                subject = pMCI_baseline_df.loc[idx, 'participant_id']
                MCI_df.drop(subject, inplace=True, level=0)
            supplementary_diagnoses.append('pMCI')

            print('Removed %i subjects' % len(pMCI_baseline_df))
            sub_df = MCI_df.reset_index().groupby('participant_id')['session_id'].nunique()
            print('%i subjects, %i scans' % (len(sub_df), len(MCI_df)))

        if len(supplementary_diagnoses) == 0:
            raise ValueError('The MCI_sub_categories flag is not needed as there are no intersections with'
                             'MCI subcategories.')

        diagnosis_baseline_df = baseline_df(MCI_df, 'MCI', False)
        diagnoses_list = list(diagnosis_baseline_df.diagnosis)
        unique = list(set(diagnoses_list))
        y = np.array(
            [unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

        splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):

            train_index, test_index = indices

            test_df = diagnosis_baseline_df.iloc[test_index]
            train_df = diagnosis_baseline_df.iloc[train_index]

            # Add the sub categories
            for diagnosis in supplementary_diagnoses:
                sup_train_df = pd.read_csv(path.join(train_path, 'split-' + str(i), str(diagnosis) + '_baseline.tsv'),
                                           sep='\t')
                train_df = pd.concat([train_df, sup_train_df])
                sup_test_df = pd.read_csv(path.join(test_path, 'split-' + str(i), str(diagnosis) + '_baseline.tsv'),
                                          sep='\t')
                test_df = pd.concat([test_df, sup_test_df])

            train_df.reset_index(inplace=True, drop=True)
            test_df.reset_index(inplace=True, drop=True)
            train_df.diagnosis = ['MCI'] * len(train_df)
            test_df.diagnosis = ['MCI'] * len(test_df)

            # Retrieve all sessions for the training set
            complete_train_df = pd.DataFrame()
            for idx in train_df.index.values:
                subject = train_df.loc[idx, 'participant_id']
                subject_df = diagnosis_df[diagnosis_df.participant_id == subject]
                complete_train_df = pd.concat([complete_train_df, subject_df])

            complete_train_df.to_csv(path.join(train_path, 'split-' + str(i), 'MCI.tsv'),
                                     sep='\t', index=False)
            train_df.to_csv(
                path.join(train_path, 'split-' + str(i), 'MCI_baseline.tsv'), sep='\t', index=False)
            test_df.to_csv(
                path.join(test_path, 'split-' + str(i), 'MCI_baseline.tsv'), sep='\t', index=False)
