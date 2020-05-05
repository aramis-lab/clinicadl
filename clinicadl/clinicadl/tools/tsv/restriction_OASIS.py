# coding: utf8

import argparse
import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("merged_tsv", type=str,
                        help="Path to the file obtained by the command clinica iotools merge-tsv.")
    parser.add_argument("results_path", type=str,
                        help="Path to the resulting tsv file (filename included).")

    # Modality selection
    parser.add_argument("--age_restriction", type=int, default=62,
                        help="Includes all the subjects older than the limit age (limit age included).")

    args = parser.parse_args()

    merged_df = pd.read_csv(args.merged_tsv, sep='\t')
    results_df = merged_df[merged_df.age_bl >= args.age_restriction]
    results_df.to_csv(args.results_path, sep='\t', index=False)
