# coding: utf-8

__author__ = "Elina Thibeau--Sutre"
__copyright__ = "Copyright 2016-2019 The Aramis Lab Team"
__credits__ = [""]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Elina Thibeau--Sutre"
__email__ = "elina.ts@free.fr"
__status__ = "Completed"

import pandas as pd
from .tsv_utils import first_session, next_session, add_demographics
import os
from os import path
import numpy as np


def demographics_analysis(merged_tsv, formatted_data_path, results_path,
                          diagnoses, mmse_name="MMS", age_name="age", baseline=False):
    """
    Produces a tsv file with rows corresponding to the labels defined by the diagnoses list,
    and the columns being demographic statistics.

    Args:
        merged_tsv (str): Path to the file obtained by the command clinica iotools merge-tsv.
        formatted_data_path (str): Path to the folder containing data extracted by clinicadl tsvtool getlabels.
        results_path (str): Path to the output tsv file (filename included).
        diagnoses (list): Labels selected for the demographic analysis.
        mmse_name (str): Name of the variable related to the MMSE score in the merged_tsv file.
        age_name (str): Name of the variable related to the age in the merged_tsv file.
        baseline (bool): Performs the analysis based on <label>_baseline.tsv files.

    Returns:
        writes one tsv file at results_path containing the
        demographic analysis of the tsv files in formatted_data_path.
    """

    merged_df = pd.read_csv(merged_tsv, sep='\t')
    merged_df.set_index(['participant_id', 'session_id'], inplace=True)
    parent_directory = path.abspath(path.join(results_path, os.pardir))
    if not path.exists(parent_directory):
        os.makedirs(parent_directory)

    fields_dict = {'age': age_name, 'sex': 'sex', 'MMSE': mmse_name, 'CDR': 'cdr_global'}

    if age_name not in merged_df.columns.values:
        raise ValueError("The column corresponding to age is not labelled as %s. "
                         "Please change the parameters." % age_name)

    if mmse_name not in merged_df.columns.values:
        raise ValueError("The column corresponding to mmse is not labelled as %s. "
                         "Please change the parameters." % mmse_name)

    columns = ['n_subjects', 'mean_age', 'std_age',
               'min_age', 'max_age', 'sexF',
               'sexM', 'mean_MMSE', 'std_MMSE',
               'min_MMSE', 'max_MMSE', 'CDR_0',
               'CDR_0.5', 'CDR_1', 'CDR_2',
               'CDR_3', 'mean_scans', 'std_scans',
               'n_scans']
    results_df = pd.DataFrame(index=diagnoses, columns=columns, data=np.zeros((len(diagnoses), len(columns))))

    # Need all values for mean and variance (age and MMSE)
    diagnosis_dict = dict.fromkeys(diagnoses)
    for diagnosis in diagnoses:
        diagnosis_dict[diagnosis] = {'age': [], 'MMSE': [], 'scans': []}
        if baseline:
            diagnosis_path = path.join(formatted_data_path, diagnosis + '_baseline.tsv')
        else:
            diagnosis_path = path.join(formatted_data_path, diagnosis + '.tsv')
        diagnosis_df = pd.read_csv(diagnosis_path, sep='\t')
        diagnosis_demographics_df = add_demographics(diagnosis_df, merged_df, diagnosis)
        diagnosis_demographics_df.set_index(['participant_id', 'session_id'], inplace=True)
        diagnosis_df.set_index(['participant_id', 'session_id'], inplace=True)

        for subject, subject_df in diagnosis_df.groupby(level=0):
            first_session_id = first_session(subject_df)
            feature_absence = isinstance(merged_df.loc[(subject, first_session_id), 'diagnosis'], float)
            while feature_absence:
                print(subject, first_session_id)
                first_session_id = next_session(subject_df, first_session_id)
                feature_absence = isinstance(merged_df.loc[(subject, first_session_id), 'diagnosis'], float)
            demographics_subject_df = merged_df.loc[subject]

            # Extract features
            results_df.loc[diagnosis, 'n_subjects'] += 1
            results_df.loc[diagnosis, 'n_scans'] += len(subject_df)
            diagnosis_dict[diagnosis]['age'].append(
                merged_df.loc[(subject, first_session_id), fields_dict['age']])
            diagnosis_dict[diagnosis]['MMSE'].append(
                merged_df.loc[(subject, first_session_id), fields_dict['MMSE']])
            diagnosis_dict[diagnosis]['scans'].append(len(subject_df))
            sexF = len(demographics_subject_df[(demographics_subject_df[fields_dict['sex']].isin(['F']))]) > 0
            sexM = len(demographics_subject_df[(demographics_subject_df[fields_dict['sex']].isin(['M']))]) > 0
            if sexF:
                results_df.loc[diagnosis, 'sexF'] += 1
            elif sexM:
                results_df.loc[diagnosis, 'sexM'] += 1
            else:
                raise ValueError('Patient %s has no sex' % subject)

            cdr = merged_df.at[(subject, first_session_id), fields_dict['CDR']]
            if cdr == 0:
                results_df.loc[diagnosis, 'CDR_0'] += 1
            elif cdr == 0.5:
                results_df.loc[diagnosis, 'CDR_0.5'] += 1
            elif cdr == 1:
                results_df.loc[diagnosis, 'CDR_1'] += 1
            elif cdr == 2:
                results_df.loc[diagnosis, 'CDR_2'] += 1
            elif cdr == 3:
                results_df.loc[diagnosis, 'CDR_3'] += 1
            else:
                raise ValueError('Patient %s has CDR %f' % (subject, cdr))

    for diagnosis in diagnoses:
        results_df.loc[diagnosis, 'mean_age'] = np.nanmean(diagnosis_dict[diagnosis]['age'])
        results_df.loc[diagnosis, 'std_age'] = np.nanstd(diagnosis_dict[diagnosis]['age'])
        results_df.loc[diagnosis, 'min_age'] = np.min(diagnosis_dict[diagnosis]['age'])
        results_df.loc[diagnosis, 'max_age'] = np.max(diagnosis_dict[diagnosis]['age'])
        results_df.loc[diagnosis, 'mean_MMSE'] = np.nanmean(diagnosis_dict[diagnosis]['MMSE'])
        results_df.loc[diagnosis, 'std_MMSE'] = np.nanstd(diagnosis_dict[diagnosis]['MMSE'])
        results_df.loc[diagnosis, 'min_MMSE'] = np.nanmin(diagnosis_dict[diagnosis]['MMSE'])
        results_df.loc[diagnosis, 'max_MMSE'] = np.nanmax(diagnosis_dict[diagnosis]['MMSE'])
        results_df.loc[diagnosis, 'mean_scans'] = np.nanmean(diagnosis_dict[diagnosis]['scans'])
        results_df.loc[diagnosis, 'std_scans'] = np.nanstd(diagnosis_dict[diagnosis]['scans'])

    results_df.index.name = "diagnosis"

    results_df.to_csv(results_path, sep='\t')
