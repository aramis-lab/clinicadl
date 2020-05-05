# coding: utf8

import argparse
import pandas as pd
from copy import deepcopy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("merged_tsv", type=str,
                        help="Path to the file obtained by the command clinica iotools merge-tsv.")
    parser.add_argument("results_path", type=str,
                        help="Path to the resulting tsv file (filename included).")

    args = parser.parse_args()

    merged_df = pd.read_csv(args.merged_tsv, sep='\t')
    merged_df.set_index(['participant_id', 'session_id'], inplace=True)
    results_df = deepcopy(merged_df)
    for subject, subject_df in merged_df.groupby(level=0):
        examination_df = subject_df['examination_date']
        dates_list = examination_df.values
        if (dates_list == '-4').all():
            results_df.drop(subject, inplace=True)

    results_df.to_csv(args.results_path, sep='\t')
