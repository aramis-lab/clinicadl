import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def subject_diagnosis_df(subject_session_df):
    """
    Creates a DataFrame with only one occurence of each subject and the most early diagnosis
    Some subjects may not have the baseline diagnosis (ses-M00 doesn't exist)

    :param subject_session_df: (DataFrame) a DataFrame with columns containing 'participant_id', 'session_id', 'diagnosis'
    :return: DataFrame with the same columns as the input
    """
    temp_df = subject_session_df.set_index(['participant_id', 'session_id'])
    subjects_df = pd.DataFrame(columns=subject_session_df.columns)
    for subject, subject_df in temp_df.groupby(level=0):
        session_nb_list = [int(session[5::]) for _, session in subject_df.index.values]
        session_nb_list.sort()
        session_baseline_nb = session_nb_list[0]
        if session_baseline_nb < 10:
            session_baseline = 'ses-M0' + str(session_baseline_nb)
        else:
            session_baseline = 'ses-M' + str(session_baseline_nb)
        row_baseline = list(subject_df.loc[(subject, session_baseline)])
        row_baseline.insert(0, subject)
        row_baseline.insert(1, session_baseline)
        row_baseline = np.array(row_baseline).reshape(1, len(row_baseline))
        row_df = pd.DataFrame(row_baseline, columns=subject_session_df.columns)
        subjects_df = subjects_df.append(row_df)

    subjects_df.reset_index(inplace=True, drop=True)
    return subjects_df


def multiple_time_points(df, subset_df):
    """
    Returns a DataFrame with all the time points of each subject

    :param df: (DataFrame) the reference containing all the time points of all subjects.
    :param subset_df: (DataFrame) the DataFrame containing the subset of subjects.
    :return: mtp_df (DataFrame) a DataFrame with the time points of the subjects of subset_df
    """
    mtp_df = pd.DataFrame(columns=df.columns)
    temp_df = df.set_index('participant_id')
    for idx in subset_df.index.values:
        subject = subset_df.loc[idx, 'participant_id']
        subject_df = temp_df.loc[subject]
        if isinstance(subject_df, pd.Series):
            subject_id = subject_df.name
            row = list(subject_df.values)
            row.insert(0, subject_id)
            subject_df = pd.DataFrame(np.array(row).reshape(1, len(row)), columns=df.columns)
            mtp_df = mtp_df.append(subject_df)
        else:
            mtp_df = mtp_df.append(subject_df.reset_index())

    mtp_df.reset_index(inplace=True, drop=True)
    return mtp_df


def split_subjects_to_tsv(diagnoses_tsv, n_splits=5, val_size=0.15):
    """
    Write the tsv files corresponding to the train/val/test splits of all folds

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param n_splits: (int) the number of splits wanted in the cross-validation
    :param val_size: (float) proportion of the train set being used for validation
    :return: None
    """

    df = pd.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(df.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    # Here we reduce the DataFrame to have only one diagnosis per subject (multiple time points case)
    diagnosis_df = subject_diagnosis_df(df)
    diagnoses_list = list(diagnosis_df.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])  # There is one label per diagnosis depending on the order

    splits = StratifiedKFold(n_splits=n_splits, shuffle=True)
    sets_dir = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0])
    if not path.exists(sets_dir):
        os.makedirs(sets_dir)

    n_iteration = 0
    for train_index, test_index in splits.split(np.zeros(len(y)), y):
        y_train = y[train_index]
        diagnosis_df_train = diagnosis_df.loc[train_index]

        # split the train data into training and validation set
        skf_2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
        indices = next(skf_2.split(np.zeros(len(y_train)), y_train))
        train_ind, valid_ind = indices

        # We use only one session per subject in the test set
        df_test = diagnosis_df.iloc[test_index]

        df_sub_valid = diagnosis_df_train.iloc[valid_ind]
        df_sub_train = diagnosis_df_train.iloc[train_ind]
        df_valid = multiple_time_points(df, df_sub_valid)
        df_train = multiple_time_points(df, df_sub_train)

        df_train.to_csv(path.join(sets_dir, path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' +
                                  str(n_iteration) + '_train.tsv'), sep='\t', index=False)
        df_test.to_csv(path.join(sets_dir, path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' +
                                 str(n_iteration) + '_test.tsv'), sep='\t', index=False)
        df_valid.to_csv(path.join(sets_dir, path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' +
                                  str(n_iteration) + '_valid.tsv'), sep='\t', index=False)
        n_iteration += 1


def load_split(diagnoses_tsv, fold):
    """
    Returns the paths of the TSV files for each set

    :param diagnoses_tsv: (str) path to the tsv file with diagnoses
    :param fold: (int) the number of the current fold
    :return: 3 Strings
        training_tsv
        test_tsv
        valid_tsv
    """
    sets_dir = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0])

    training_tsv = path.join(sets_dir,
                             path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_train.tsv')
    test_tsv = path.join(sets_dir,
                         path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_test.tsv')
    valid_tsv = path.join(sets_dir,
                          path.basename(diagnoses_tsv).split('.')[0] + '_iteration-' + str(fold) + '_valid.tsv')

    return training_tsv, test_tsv, valid_tsv
