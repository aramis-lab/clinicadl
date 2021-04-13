# coding: utf8

from .tsv_utils import extract_baseline, retrieve_longitudinal, remove_sub_labels
from ..deep_learning.iotools import return_logger, commandline_to_json
import shutil
from sklearn.model_selection import StratifiedKFold
from os import path
import os
import pandas as pd
import numpy as np

sex_dict = {'M': 0, 'F': 1}


def write_splits(diagnosis, diagnosis_df, split_label, n_splits,
                 train_path, test_path, supplementary_diagnoses=None):
    """
    Split data at the subject-level in training and test to have equivalent distributions in split_label.

    Args:
        diagnosis: (str) diagnosis on which the split is done
        diagnosis_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        split_label: (str) label on which the split is done (categorical variables)
        n_splits: (int) Number of folds in the k-fold split
        train_path: (str) Path to the training data.
        test_path: (str) Path to the test data.
        supplementary_diagnoses: (list of str) List of supplementary diagnoses to add to the data.

    Returns:
        train_df (DataFrame) subjects in the train set
        test_df (DataFrame) subjects in the test set
    """

    baseline_df = extract_baseline(diagnosis_df, diagnosis)
    if split_label is None:
        diagnoses_list = list(baseline_df.diagnosis)
        unique = list(set(diagnoses_list))
        y = np.array([unique.index(x) for x in diagnoses_list])
    else:
        stratification_list = list(baseline_df[split_label])
        unique = list(set(stratification_list))
        y = np.array([unique.index(x) for x in stratification_list])

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

    for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):
        print(f'Split {i}')
        train_index, test_index = indices

        test_df = baseline_df.iloc[test_index]
        train_df = baseline_df.iloc[train_index]

        if supplementary_diagnoses is not None:
            for supplementary_diagnosis in supplementary_diagnoses:
                sup_train_df = pd.read_csv(path.join(train_path, f'split-{i}',
                                                     f'{supplementary_diagnosis}_baseline.tsv'), sep='\t')
                train_df = pd.concat([train_df, sup_train_df])
                sup_test_df = pd.read_csv(path.join(test_path, f'split-{i}',
                                                    f'{supplementary_diagnosis}_baseline.tsv'), sep='\t')
                test_df = pd.concat([test_df, sup_test_df])

            train_df.reset_index(inplace=True, drop=True)
            test_df.reset_index(inplace=True, drop=True)

        train_df.to_csv(path.join(train_path, f'split-{i}', f'{diagnosis}_baseline.tsv'), sep='\t', index=False)
        test_df.to_csv(path.join(test_path, f'split-{i}', f'{diagnosis}_baseline.tsv'), sep='\t', index=False)

        long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
        long_train_df.to_csv(path.join(train_path, f'split-{i}', f'{diagnosis}.tsv'), sep='\t', index=False)


def split_diagnoses(formatted_data_path,
                    n_splits=5, subset_name="validation", MCI_sub_categories=True,
                    stratification=None, verbose=0):
    """
    Performs a k-fold split for each label independently on the subject level.
    The train folder will contain two lists per fold per diagnosis (baseline and longitudinal),
    whereas the test folder will only include the list of baseline sessions for each fold.

    Args:
        formatted_data_path (str): Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        n_splits (int): Number of folds in the k-fold split.
        subset_name (str): Name of the subset that is complementary to train.
        MCI_sub_categories (bool): If True, manages MCI sub-categories to avoid data leakage.
        stratification (str): Name of variable used to stratify k-fold.
        verbose (int): level of verbosity.

    Returns:
         writes three files per split per <label>.tsv file present in formatted_data_path:
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>.tsv
            - formatted_data_path/train_splits-<n_splits>/split-<split>/<label>_baseline.tsv
            - formatted_data_path/<subset_name>_splits-<n_splits>/split-<split>/<label>_baseline.tsv
    """
    logger = return_logger(verbose, 'k-fold split')

    commandline_to_json({
        "output_dir": formatted_data_path,
        "n_splits": n_splits,
        "subset_name": subset_name,
        "MCI_sub_categories": MCI_sub_categories,
        "stratification": stratification
    }, filename="kfold.json")

    # Read files
    results_path = formatted_data_path

    train_path = path.join(results_path, f'train_splits-{n_splits}')
    if path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path)
    for i in range(n_splits):
        os.mkdir(path.join(train_path, f'split-{i}'))

    test_path = path.join(results_path, f'{subset_name}_splits-{n_splits}')
    if path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)
    for i in range(n_splits):
        os.mkdir(path.join(test_path, f'split-{i}'))

    diagnosis_df_paths = os.listdir(results_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.tsv')]
    diagnosis_df_paths = [x for x in diagnosis_df_paths if not x.endswith('_baseline.tsv')]

    MCI_special_treatment = False

    if 'MCI.tsv' in diagnosis_df_paths:
        if MCI_sub_categories:
            diagnosis_df_paths.remove('MCI.tsv')
            MCI_special_treatment = True
        elif 'sMCI.tsv' in diagnosis_df_paths or 'pMCI.tsv' in diagnosis_df_paths:
            logger.warning("MCI special treatment was deactivated though MCI subgroups were found. "
                           "Be aware that it may cause data leakage in transfer learning tasks.")

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis_df_path in diagnosis_df_paths:
        diagnosis = diagnosis_df_path.split('.')[0]

        diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path), sep='\t')
        write_splits(diagnosis, diagnosis_df, stratification, n_splits, train_path, test_path)

        logger.info(f"K-fold split for diagnosis {diagnosis} is done.")

    if MCI_special_treatment:

        # Extraction of MCI subjects without intersection with the sMCI / pMCI train
        diagnosis_df = pd.read_csv(path.join(results_path, 'MCI.tsv'), sep='\t')
        MCI_df = diagnosis_df.set_index(['participant_id', 'session_id'])
        MCI_df, supplementary_diagnoses = remove_sub_labels(MCI_df, ["sMCI", "pMCI"],
                                                            diagnosis_df_paths, results_path,
                                                            logger=logger)

        if len(supplementary_diagnoses) == 0:
            raise ValueError('The MCI_sub_categories flag is not needed as there are no intersections with '
                             'MCI subcategories.')

        MCI_df.reset_index(drop=False, inplace=True)
        logger.debug(MCI_df)
        write_splits('MCI', MCI_df, stratification, n_splits, train_path, test_path,
                     supplementary_diagnoses=supplementary_diagnoses)
        logger.info("K-fold split for diagnosis MCI is done.")
