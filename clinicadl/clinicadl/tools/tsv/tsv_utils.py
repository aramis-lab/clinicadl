# coding: utf8

from copy import copy
import numpy as np
import pandas as pd
from os import path


def neighbour_session(session, session_list, neighbour):
    if session not in session_list:
        temp_list = session_list + [session]
        temp_list.sort()
    else:
        temp_list = copy(session_list)
        temp_list.sort()
    index_session = temp_list.index(session)

    if index_session + neighbour < 0 or index_session + neighbour >= len(temp_list):
        return None
    else:
        if temp_list[index_session + neighbour] < 10:
            return 'ses-M0' + str(temp_list[index_session + neighbour])
        else:
            return 'ses-M' + str(temp_list[index_session + neighbour])


def after_end_screening(session, session_list):
    if session in session_list:
        return False
    else:
        temp_list = session_list + [session]
        temp_list.sort()
        index_session = temp_list.index(session)
        return index_session == len(temp_list) - 1


def last_session(session_list):
    temp_list = copy(session_list)
    temp_list.sort()
    if temp_list[-1] < 10:
        return 'ses-M0' + str(temp_list[-1])
    else:
        return 'ses-M' + str(temp_list[-1])


def complementary_list(total_list, sub_list):
    result_list = []
    for element in total_list:
        if element not in sub_list:
            result_list.append(element)
    return result_list


def first_session(subject_df):
    session_list = [int(session[5:]) for _, session in subject_df.index.values]
    session_list.sort()
    first_session = session_list[0]
    if first_session < 10:
        return 'ses-M0' + str(first_session)
    else:
        return 'ses-M' + str(first_session)


def next_session(subject_df, session_orig):
        session_list = [int(session[5:]) for _, session in subject_df.index.values]
        session_list.sort()
        session_id_list = []
        for session in session_list:
            if session < 10:
                session_id_list.append('ses-M0' + str(session))
            else:
                session_id_list.append('ses-M' + str(session))
        index = session_id_list.index(session_orig)
        if index < len(session_id_list) - 1:
            return session_id_list[index + 1]
        else:
            raise ValueError('The argument session is the last session')


def extract_baseline(diagnosis_df, diagnosis, set_index=True):
    from copy import deepcopy

    if set_index:
        all_df = diagnosis_df.set_index(['participant_id', 'session_id'])
    else:
        all_df = deepcopy(diagnosis_df)

    result_df = pd.DataFrame()
    for subject, subject_df in all_df.groupby(level=0):
        baseline = first_session(subject_df)
        subject_baseline_df = pd.DataFrame(data=[[subject, baseline] +
                                                 subject_df.loc[(subject, baseline)].tolist()],
                                           columns=["participant_id", "session_id"] + subject_df.columns.values.tolist())
        result_df = pd.concat([result_df, subject_baseline_df])

    result_df["diagnosis"] = [diagnosis] * len(result_df)
    result_df.reset_index(inplace=True, drop=True)

    return result_df


def chi2(x_test, x_train):
    from scipy.stats import chisquare

    # Look for chi2 computation
    total_categories = np.concatenate([x_test, x_train])
    unique_categories = np.unique(total_categories)
    f_obs = [(x_test == category).sum() / len(x_test) for category in unique_categories]
    f_exp = [(x_train == category).sum() / len(x_train) for category in unique_categories]

    T, p = chisquare(f_obs, f_exp)

    return T, p


def add_demographics(df, demographics_df, diagnosis):
    out_df = pd.DataFrame()
    tmp_demo_df = copy(demographics_df)
    tmp_demo_df.reset_index(inplace=True)
    for idx in df.index.values:
        participant = df.loc[idx, "participant_id"]
        session = df.loc[idx, "session_id"]
        row_df = tmp_demo_df[(tmp_demo_df.participant_id == participant) & (tmp_demo_df.session_id == session)]
        out_df = pd.concat([out_df, row_df])
    out_df.reset_index(inplace=True, drop=True)
    out_df.diagnosis = [diagnosis] * len(out_df)
    return out_df


def remove_unicity(values_list):
    """Count the values of each class and label all the classes with only one label under the same label."""
    unique_classes, counts = np.unique(values_list, return_counts=True)
    one_sub_classes = unique_classes[(counts == 1)]
    for class_element in one_sub_classes:
        values_list[values_list.index(class_element)] = unique_classes.min()

    return values_list


def category_conversion(values_list):
    values_np = np.array(values_list)
    unique_classes = np.unique(values_np)
    for index, unique_class in enumerate(unique_classes):
        values_np[values_np == unique_class] = index + 1

    return values_np.astype(int).tolist()


def find_label(labels_list, target_label):
    if target_label in labels_list:
        return target_label
    else:
        min_length = np.inf
        found_label = None
        for label in labels_list:
            if target_label.lower() in label.lower() and min_length > len(label):
                min_length = len(label)
                found_label = label
        if found_label is None:
            raise ValueError(f"No label was found in {labels_list} for target label {target_label}.")

        return found_label


def retrieve_longitudinal(df, diagnosis_df):
    final_df = pd.DataFrame()
    for idx in df.index.values:
        subject = df.loc[idx, 'participant_id']
        row_df = diagnosis_df[diagnosis_df.participant_id == subject]
        final_df = pd.concat([final_df, row_df])

    return final_df


def remove_sub_labels(diagnosis_df, sub_labels, diagnosis_df_paths, results_path,
                      logger=None):

    from ..deep_learning.iotools import return_logger

    if logger is None:
        logger = return_logger(2, "remove sub labels")

    supplementary_diagnoses = []

    logger.debug('Before subjects removal')
    sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
    logger.debug(f'{len(sub_df)} subjects, {len(diagnosis_df)} scans')

    for label in sub_labels:
        if f'{label}.tsv' in diagnosis_df_paths:
            sub_diag_df = pd.read_csv(path.join(results_path, f'{label}.tsv'), sep='\t')
            sub_diag_baseline_df = extract_baseline(sub_diag_df, label)
            for idx in sub_diag_baseline_df.index.values:
                subject = sub_diag_baseline_df.loc[idx, 'participant_id']
                diagnosis_df.drop(subject, inplace=True, level=0)
            supplementary_diagnoses.append(label)

            logger.debug(f'Removed {len(sub_diag_baseline_df)} subjects based on {label} label')
            sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
            logger.debug(f'{len(sub_df)} subjects, {len(diagnosis_df)} scans')

    return diagnosis_df, supplementary_diagnoses
