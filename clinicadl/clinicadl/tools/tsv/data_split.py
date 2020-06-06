# coding: utf8

from .tsv_utils import complementary_list, add_demographics, baseline_df, chi2
from scipy.stats import ttest_ind
import shutil
import pandas as pd
from os import path
import numpy as np
import os

sex_dict = {'M': 0, 'F': 1}


def create_split(diagnosis, diagnosis_df, merged_df, n_test, age_name="age",
                 pval_threshold_ttest=0.80, t_val_chi2_threshold=0.0642):
    """
    Split data at the subject-level in training and test set with equivalent age and sex distributions

    :param diagnosis: (str) diagnosis on which the split is done
    :param diagnosis_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
    :param merged_df: DataFrame with columns including ['age', 'sex'] and containing the same sessions as diagnosis_df
    :param n_test: (float)
        If > 1 number of subjects to put in the test set.
        If < 1 proportion of subjects to put in the test set.
    :param age_name: (str) label of the age column in the dataset.
    :param pval_threshold_ttest: (float) threshold for the t-test on age
    :param t_val_chi2_threshold:  (float) threshold for the chi2 test on sex
    :return:
        train_df (DataFrame) subjects in the train set
        test_df (DataFrame) subjects in the test set
    """

    diagnosis_baseline_df = baseline_df(diagnosis_df, diagnosis)
    baseline_demographics_df = add_demographics(diagnosis_baseline_df, merged_df, diagnosis)

    if n_test > 1:
        n_test = int(n_test)
    else:
        n_test = int(n_test * len(diagnosis_baseline_df))

    sex = list(baseline_demographics_df.sex.values)
    age = list(baseline_demographics_df[age_name].values)

    idx = np.arange(len(diagnosis_baseline_df))

    flag_selection = True
    n_try = 0

    while flag_selection:
        idx_test = np.random.choice(idx, size=n_test, replace=False)
        idx_test.sort()
        idx_train = complementary_list(idx, idx_test)

        # Find the value for different demographical values (age, MMSE, sex)
        age_test = [float(age[idx]) for idx in idx_test]
        age_train = [float(age[idx]) for idx in idx_train]

        sex_test = [sex_dict[sex[idx]] for idx in idx_test]
        sex_train = [sex_dict[sex[idx]] for idx in idx_train]

        t_age, p_age = ttest_ind(age_test, age_train)
        T_sex = chi2(sex_test, sex_train)

        if T_sex < t_val_chi2_threshold and p_age > pval_threshold_ttest:
            flag_selection = False
            test_df = baseline_demographics_df.loc[idx_test]
            train_df = baseline_demographics_df.loc[idx_train]

        n_try += 1

    print("Split for diagnosis %s was found after %i trials" % (diagnosis, n_try))
    return train_df, test_df


def split_diagnoses(merged_tsv, formatted_data_path,
                    n_test=100, age_name="age", subset_name="test", MCI_sub_categories=True,
                    t_val_threshold=0.0642, p_val_threshold=0.80):
    """
    Performs a single split for each label independently on the subject level.
    The train folder will contain two lists per diagnosis (baseline and longitudinal),
    whereas the test folder will only include the list of baseline sessions.

    The age and sex distributions between the two sets must be non-significant (according to T-test and chi-square).

    Args:
        merged_tsv (str): Path to the file obtained by the command clinica iotools merge-tsv.
        formatted_data_path (str): Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        n_test (float):
            If > 1, number of subjects to put in set with name 'subset_name'.
            If < 1, proportion of subjects to put in set with name 'subset_name'.
            If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.
        age_name (str): Label of the age column in the dataset.
        subset_name (str): Name of the subset that is complementary to train.
        MCI_sub_categories (bool): If True, manages MCI sub-categories to avoid data leakage.
        t_val_threshold (float): The threshold used for the chi2 test on sex distributions.
        p_val_threshold (float): The threshold used for the T-test on age distributions.

    Returns:
        writes three files per <label>.tsv file present in formatted_data_path:
            - formatted_data_path/train/<label>.tsv
            - formatted_data_path/train/<label>_baseline.tsv
            - formatted_data_path/<subset_name>/<label>_baseline.tsv
    """
    # Read files
    merged_df = pd.read_csv(merged_tsv, sep='\t')
    merged_df.set_index(['participant_id', 'session_id'], inplace=True)
    results_path = formatted_data_path

    train_path = path.join(results_path, 'train')
    if path.exists(train_path):
        shutil.rmtree(train_path)
    if n_test > 0:
        os.makedirs(train_path)

    test_path = path.join(results_path, subset_name)
    if path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)

    diagnosis_df_paths = os.listdir(results_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.tsv')]
    diagnosis_df_paths = [x for x in diagnosis_df_paths if not x.endswith('_baseline.tsv')]

    MCI_special_treatment = False

    if MCI_sub_categories and 'MCI.tsv' in diagnosis_df_paths and n_test > 0:
        diagnosis_df_paths.remove('MCI.tsv')
        MCI_special_treatment = True

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis_df_path in diagnosis_df_paths:
        print(diagnosis_df_path)
        diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path),
                                   sep='\t')
        diagnosis = diagnosis_df_path.split('.')[0]
        if n_test > 0:
            train_df, test_df = create_split(diagnosis, diagnosis_df, merged_df, age_name=age_name,
                                             n_test=n_test, t_val_chi2_threshold=t_val_threshold,
                                             pval_threshold_ttest=p_val_threshold)
            # Save baseline splits
            train_df = train_df[['participant_id', 'session_id', 'diagnosis']]
            train_df.to_csv(path.join(train_path, str(diagnosis) + '_baseline.tsv'), sep='\t', index=False)
            test_df = test_df[['participant_id', 'session_id', 'diagnosis']]
            test_df.to_csv(path.join(test_path, str(diagnosis) + '_baseline.tsv'), sep='\t', index=False)

            # Retrieve all sessions for the training set
            complete_train_df = pd.DataFrame()
            for idx in train_df.index.values:
                subject = train_df.loc[idx, 'participant_id']
                subject_df = diagnosis_df[diagnosis_df.participant_id == subject]
                complete_train_df = pd.concat([complete_train_df, subject_df])

            complete_train_df.to_csv(path.join(train_path, str(diagnosis) + '.tsv'), sep='\t', index=False)

        else:
            diagnosis_baseline_df = baseline_df(diagnosis_df, diagnosis)
            test_df = diagnosis_baseline_df[['participant_id', 'session_id', 'diagnosis']]
            test_df.to_csv(path.join(test_path, str(diagnosis) + '_baseline.tsv'), sep='\t', index=False)

    if MCI_special_treatment:

        # Extraction of MCI subjects without intersection with the sMCI / pMCI train
        diagnosis_df = pd.read_csv(path.join(results_path, 'MCI.tsv'), sep='\t')
        MCI_df = diagnosis_df.set_index(['participant_id', 'session_id'])
        supplementary_diagnoses = []

        print('Before subjects removal')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        print('%i subjects, %i scans' % (len(sub_df), len(diagnosis_df)))

        if 'sMCI.tsv' in diagnosis_df_paths:
            sMCI_baseline_train_df = pd.read_csv(path.join(train_path, 'sMCI_baseline.tsv'), sep='\t')
            sMCI_baseline_test_df = pd.read_csv(path.join(test_path, 'sMCI_baseline.tsv'), sep='\t')
            sMCI_baseline_df = pd.concat([sMCI_baseline_train_df, sMCI_baseline_test_df])
            sMCI_baseline_df.reset_index(drop=True, inplace=True)
            for idx in sMCI_baseline_df.index.values:
                subject = sMCI_baseline_df.loc[idx, 'participant_id']
                MCI_df.drop(subject, inplace=True)
            supplementary_diagnoses.append('sMCI')

            print('Removed %i subjects' % len(sMCI_baseline_df))
            sub_df = MCI_df.reset_index().groupby('participant_id')['session_id'].nunique()
            print('%i subjects, %i scans' % (len(sub_df), len(MCI_df)))

        if 'pMCI.tsv' in diagnosis_df_paths:
            pMCI_baseline_train_df = pd.read_csv(path.join(train_path, 'pMCI_baseline.tsv'), sep='\t')
            pMCI_baseline_test_df = pd.read_csv(path.join(test_path, 'pMCI_baseline.tsv'), sep='\t')
            pMCI_baseline_df = pd.concat([pMCI_baseline_train_df, pMCI_baseline_test_df])
            pMCI_baseline_df.reset_index(drop=True, inplace=True)
            for idx in pMCI_baseline_df.index.values:
                subject = pMCI_baseline_df.loc[idx, 'participant_id']
                MCI_df.drop(subject, inplace=True)
            supplementary_diagnoses.append('pMCI')

            print('Removed %i subjects' % len(pMCI_baseline_df))
            sub_df = MCI_df.reset_index().groupby('participant_id')['session_id'].nunique()
            print('%i subjects, %i scans' % (len(sub_df), len(MCI_df)))

        if len(supplementary_diagnoses) == 0:
            raise ValueError('The MCI_sub_categories flag is not needed as there are no intersections with'
                             'MCI subcategories.')

        # Construction of supplementary train
        supplementary_train_df = pd.DataFrame()
        for diagnosis in supplementary_diagnoses:
            sup_baseline_train_df = pd.read_csv(path.join(train_path, diagnosis + '_baseline.tsv'), sep='\t')
            supplementary_train_df = pd.concat([supplementary_train_df, sup_baseline_train_df])
            sub_df = supplementary_train_df.reset_index().groupby('participant_id')['session_id'].nunique()
            print('supplementary_train_df %i subjects, %i scans' % (len(sub_df), len(supplementary_train_df)))

        supplementary_train_df.reset_index(drop=True, inplace=True)
        supplementary_train_df = add_demographics(supplementary_train_df, merged_df, 'MCI')

        # MCI selection
        MCI_df.reset_index(inplace=True)
        diagnosis_baseline_df = baseline_df(MCI_df, 'MCI')
        baseline_demographics_df = add_demographics(diagnosis_baseline_df, merged_df, 'MCI')
        complete_diagnosis_baseline_df = baseline_df(diagnosis_df, 'MCI')

        if n_test > 1:
            n_test = int(n_test)
        else:
            n_test = int(n_test * len(complete_diagnosis_baseline_df))

        sex = list(baseline_demographics_df.sex.values)
        age = list(baseline_demographics_df[age_name].values)

        sup_train_sex = list(supplementary_train_df.sex.values)
        sup_train_age = list(supplementary_train_df[age_name].values)

        sup_train_sex = [sex_dict[x] for x in sup_train_sex]
        sup_train_age = [float(x) for x in sup_train_age]

        idx = np.arange(len(diagnosis_baseline_df))

        flag_selection = True
        n_try = 0

        while flag_selection:
            idx_test = np.random.choice(idx, size=n_test, replace=False)
            idx_test.sort()
            idx_train = complementary_list(idx, idx_test)

            # Find the value for different demographical values (age, MMSE, sex)
            age_test = [float(age[idx]) for idx in idx_test]
            age_train = [float(age[idx]) for idx in idx_train] + sup_train_age

            sex_test = [sex_dict[sex[idx]] for idx in idx_test]
            sex_train = [sex_dict[sex[idx]] for idx in idx_train] + sup_train_sex

            t_age, p_age = ttest_ind(age_test, age_train)
            T_sex = chi2(sex_test, sex_train)

            if T_sex < t_val_threshold and p_age > p_val_threshold:
                flag_selection = False
                MCI_baseline_test_df = baseline_demographics_df.loc[idx_test]
                train_df = baseline_demographics_df.loc[idx_train]
                MCI_baseline_train_df = pd.concat([train_df, supplementary_train_df])
                print('Supplementary train df', len(supplementary_train_df))
                MCI_baseline_train_df.reset_index(drop=True, inplace=True)

            n_try += 1

        print('Split for diagnosis MCI was found after %i trials' % n_try)

        # Write selection of MCI
        MCI_baseline_train_df = MCI_baseline_train_df[['participant_id', 'session_id', 'diagnosis']]
        MCI_baseline_train_df.to_csv(path.join(train_path, 'MCI_baseline.tsv'), sep='\t', index=False)
        MCI_baseline_test_df = MCI_baseline_test_df[['participant_id', 'session_id', 'diagnosis']]
        MCI_baseline_test_df.to_csv(path.join(test_path, 'MCI_baseline.tsv'), sep='\t', index=False)

        # Retrieve all sessions for the training set
        MCI_complete_train_df = pd.DataFrame()
        for idx in MCI_baseline_train_df.index.values:
            subject = MCI_baseline_train_df.loc[idx, 'participant_id']
            subject_df = diagnosis_df[diagnosis_df.participant_id == subject]
            MCI_complete_train_df = pd.concat([MCI_complete_train_df, subject_df])

        MCI_complete_train_df.to_csv(path.join(train_path, 'MCI.tsv'), sep='\t', index=False)
