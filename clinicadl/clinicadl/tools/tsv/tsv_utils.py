# coding: utf8

from copy import copy
import numpy as np
import pandas as pd


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


def baseline_df(diagnosis_df, diagnosis, set_index=True):
    from copy import deepcopy

    if set_index:
        all_df = diagnosis_df.set_index(['participant_id', 'session_id'])
    else:
        all_df = deepcopy(diagnosis_df)
    columns = ['participant_id', 'session_id', 'diagnosis']
    result_df = pd.DataFrame()
    for subject, subject_df in all_df.groupby(level=0):
        first_session_id = first_session(subject_df)
        data = np.array([subject, first_session_id, diagnosis]).reshape(1, 3)
        subject_baseline_df = pd.DataFrame(data, columns=columns)
        result_df = pd.concat([result_df, subject_baseline_df])

    result_df.reset_index(inplace=True, drop=True)

    return result_df


def chi2(x_test, x_train):
    # Look for chi2 computation
    p_expectedF = np.sum(x_train) / len(x_train)
    p_expectedM = 1 - p_expectedF

    expectedF = p_expectedF * len(x_test)
    expectedM = p_expectedM * len(x_test)
    observedF = np.sum(x_test)
    observedM = len(x_test) - np.sum(x_test)

    T = (expectedF - observedF) ** 2 / expectedF + (expectedM - observedM) ** 2 / expectedM

    return T


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
