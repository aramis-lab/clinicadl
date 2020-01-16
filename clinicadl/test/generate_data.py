"""
This file generates data for trivial or intractable (random) data for binary classification.
"""
import pandas as pd
import numpy as np
import argparse


def generate_random(caps_dir, data_df):
    """
    Generates an intractable classification task from the first subject of the tsv file

    :param caps_dir: (str) path to the CAPS directory.
    :param data_df: (DataFrame) list of subjects/sessions.
    """
    participant_id = data_df.iloc[0, 'participant_id']
    session_id = data_df.iloc[0, 'session_id']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argparser for synthetic data generation")

    parser.add_argument("caps_dir", type=str,
                        help="Data using CAPS structure. Must include the output of t1-volume-tissue-segmentation "
                             "pipeline of clinica.")
    parser.add_argument("tsv_file", type=str,
                        help="tsv file with subjets/sessions to process.")
    parser.add_argument("output_dir", type=str,
                        help="Folder containing the final dataset in CAPS format.")

    parser.add_argument("--selection", default="trivial", type=str, choices=["trivial", "random"],
                        help="Chooses which type of synthetic dataset is wanted.")

