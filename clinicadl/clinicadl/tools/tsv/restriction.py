# coding: utf8

import pandas as pd
from copy import deepcopy


def aibl_restriction(merged_tsv, results_path):
    """
    Writes a new file from which participants who do not have an examination date
    in any of their sessions are removed.
    This restriction was originally applied to AIBL dataset.

    Args:
        merged_tsv (str): Path to the file obtained by the command clinica iotools merge-tsv.
        results_path (str): Path to the output tsv file (filename included).

    Returns:
        writes a tsv file at results_path
    """
    merged_df = pd.read_csv(merged_tsv, sep='\t')
    merged_df.set_index(['participant_id', 'session_id'], inplace=True)
    results_df = deepcopy(merged_df)
    for subject, subject_df in merged_df.groupby(level=0):
        examination_df = subject_df['examination_date']
        dates_list = examination_df.values
        if (dates_list == '-4').all():
            results_df.drop(subject, inplace=True)

    results_df.to_csv(results_path, sep='\t')


def oasis_restriction(merged_tsv, results_path):
    """
    Writes a new file from which participants who are 61 years old or younger are removed.
    This restriction was originally applied to OASIS dataset

    Args:
        merged_tsv (str): Path to the file obtained by the command clinica iotools merge-tsv.
        results_path (str): Path to the output tsv file (filename included).

    Returns:
        writes a tsv file at results_path
    """
    merged_df = pd.read_csv(merged_tsv, sep='\t')
    results_df = merged_df[merged_df.age_bl >= 62]
    results_df.to_csv(results_path, sep='\t', index=False)
