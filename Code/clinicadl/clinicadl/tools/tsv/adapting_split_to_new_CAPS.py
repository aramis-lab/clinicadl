"""
We used another preprocessing with different outputs (the quality check of our previous preprocessing had excluded some
sessions).
"""


def check_baseline_existence(CAPS_path, diagnosis_df, diagnosis, enrich=False):
    """
    Select only sessions for which the modality is present

    :param CAPS_path: (str) path to the a CAPS folder
    :param diagnosis_df: (DataFrame) columns must contain ['participant_id', 'session_id', 'diagnosis']
    :return: DataFrame
    """
    from copy import deepcopy

    results_df = deepcopy(diagnosis_df)
    for idx in diagnosis_df.index:
        subject = diagnosis_df.loc[idx, 'participant_id']
        session = diagnosis_df.loc[idx, 'session_id']
        subject_path = path.join(CAPS_path, 'subjects', subject)
        session_path = path.join(CAPS_path, 'subjects', subject, session)
        if path.exists(session_path):
            pass
        elif session == 'ses-M00':
            session_path_python2 = path.join(subject_path, 'ses-M00-python2')
            if path.exists(session_path_python2):
                pass
        elif enrich and path.exists(subject_path) and "pMCI" not in diagnosis:
            baseline_path = path.join(subject_path, 'ses-M00')
            baseline_path_python2 = path.join(subject_path, 'ses-M00-python2')
            if path.exists(baseline_path) or path.exists(baseline_path_python2):
                # This could lead to label errors as pMCI may be sMCI before their baseline sessions
                results_df.loc[idx, 'session_id'] = 'ses-M00'
        else:
            results_df.drop(idx, inplace=True)

    return results_df


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from os import path
    import os

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("data_path", type=str,
                        help="Path to the tsv files after the split.")
    parser.add_argument("CAPS_path", type=str,
                        help="Path to the new CAPS.")

    # Modality selection
    parser.add_argument("--n_splits", "-s", default=None, type=int,
                        help="Number of splits to loop over all folders when needed.")
    parser.add_argument("--enrich", "-e", default=False, action="store_true",
                        help="Allow to use baseline sessions of")

    args = parser.parse_args()

    # Reading files
    if args.n_splits is not None:
        for split in range(args.n_splits):
            split_path = path.join(args.data_path, 'split-' + str(split))
            diagnoses = os.listdir(split_path)
            diagnoses = [x for x in diagnoses if x.endswith('.tsv')]

            result_path = path.join(split_path, 'SPM')
            if not path.exists(result_path):
                os.mkdir(result_path)

            for diagnosis in diagnoses:
                diagnosis_path = path.join(split_path, diagnosis)
                diagnosis_df = pd.read_csv(diagnosis_path, sep='\t')
                if diagnosis.endswith('_baseline.tsv'):
                    result_df = check_baseline_existence(args.CAPS_path, diagnosis_df, diagnosis, args.enrich)
                else:
                    result_df = check_baseline_existence(args.CAPS_path, diagnosis_df, diagnosis, False)
                result_df.to_csv(path.join(result_path, diagnosis), sep='\t', index=False)
                print(diagnosis, str(len(result_df)) + '/' + str(len(diagnosis_df)))

    else:
        diagnoses = os.listdir(args.data_path)
        diagnoses = [x for x in diagnoses if x.endswith('.tsv')]

        result_path = path.join(args.data_path, 'SPM')
        if not path.exists(result_path):
            os.mkdir(result_path)

        for diagnosis in diagnoses:
            diagnosis_path = path.join(args.data_path, diagnosis)
            diagnosis_df = pd.read_csv(diagnosis_path, sep='\t')
            result_df = check_baseline_existence(args.CAPS_path, diagnosis_df, diagnosis, args.enrich)
            result_df.to_csv(path.join(result_path, diagnosis), sep='\t', index=False)
