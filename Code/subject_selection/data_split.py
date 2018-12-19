from utils import *
from scipy.stats import ttest_ind

sex_dict = {'M': 0, 'F': 1}


def create_split(diagnosis, diagnosis_df, merged_df, n_test,
                 pval_threshold_ttest=0.80, t_val_chi2_threshold=0.0642):

    diagnosis_baseline_df = baseline_df(diagnosis_df, diagnosis)
    baseline_demographics_df = add_demographics(diagnosis_baseline_df, merged_df, diagnosis)

    sex = list(baseline_demographics_df.sex.values)
    age = list(baseline_demographics_df.age)

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

    print("Split for diagnosis %s was found after %i trials" %(diagnosis, n_try))
    return train_df, test_df


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os
    from os import path
    from copy import copy
    import numpy as np

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("merged_tsv", type=str,
                        help="Path to the file obtained by the command clinica iotools merge-tsv.")
    parser.add_argument("formatted_data_path", type=str,
                        help="Path to the folder containing formatted data.")

    # Modality selection
    parser.add_argument("--n_test", type=int, default=100,
                        help="Define the number of subjects to put in test set."
                             "If 0, there is no training set and the whole dataset is considered as a test set.")
    parser.add_argument("--tasks", nargs="+", type=str,
                        default=['AD_CN'], help="Create lists with specific tasks")
    parser.add_argument("--MCI_sub_categories", action="store_true", default=False,
                        help="Manage MCI sub-categories to avoid data leakage")

    args = parser.parse_args()

    # Read files
    merged_df = pd.read_csv(args.merged_tsv, sep='\t')
    merged_df.set_index(['participant_id', 'session_id'], inplace=True)
    results_path = path.join(args.formatted_data_path, 'lists_by_diagnosis')

    train_path = path.join(results_path, 'train')
    if not path.exists(train_path) and args.n_test > 0:
        os.makedirs(train_path)

    test_path = path.join(results_path, 'test')
    if not path.exists(test_path):
        os.makedirs(test_path)

    diagnosis_df_paths = os.listdir(results_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith('.tsv')]
    MCI_special_treatment = False

    if args.MCI_sub_categories and 'MCI.tsv' in diagnosis_df_paths:
        diagnosis_df_paths.remove('MCI.tsv')
        MCI_special_treatment = True

    # The baseline session must be kept before or we are taking all the sessions to mix them
    for diagnosis_df_path in diagnosis_df_paths:
        print('Data path', path.join(results_path, diagnosis_df_path))
        diagnosis_df = pd.read_csv(path.join(results_path, diagnosis_df_path),
                                   sep='\t')
        diagnosis = diagnosis_df_path.split('.')[0]
        if args.n_test > 0:
            train_df, test_df = create_split(diagnosis, diagnosis_df, merged_df,
                                             n_test=args.n_test)
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

    