import os
import shutil
from os import path

import pandas as pd

from clinicadl.utils.caps_dataset.data import load_data_test
from clinicadl.utils.split_manager import KFoldSplit

merged_tsv = "data/tsvtool/anonymous_BIDS.tsv"
missing_mods = "data/tsvtool/anonymous_missing_mods"
reference_path = "data/tsvtool/anonymous_reference"
diagnoses = "AD CN MCI pMCI sMCI"

"""
Check the absence of data leakage
    1) Baseline datasets contain only one scan per subject
    2) No intersection between train and test sets
    3) Absence of MCI train subjects in test sets of subcategories of MCI
"""


def check_subject_unicity(diagnosis_path):
    print("Check unicity", diagnosis_path)
    diagnosis_df_paths = os.listdir(diagnosis_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith("_baseline.tsv")]

    for diagnosis_df_path in diagnosis_df_paths:
        flag_unique = True
        check_df = pd.read_csv(path.join(diagnosis_path, diagnosis_df_path), sep="\t")
        check_df.set_index(["participant_id", "session_id"], inplace=True)
        for subject, subject_df in check_df.groupby(level=0):
            if len(subject_df) > 1:
                flag_unique = False

        assert flag_unique


def check_independance(train_path, test_path):
    print("Check independence")
    diagnosis_df_paths = os.listdir(train_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith("_baseline.tsv")]

    for diagnosis_df_path in diagnosis_df_paths:
        flag_independant = True
        train_df = pd.read_csv(path.join(train_path, diagnosis_df_path), sep="\t")
        train_df.set_index(["participant_id", "session_id"], inplace=True)
        test_df = pd.read_csv(path.join(test_path, diagnosis_df_path), sep="\t")
        test_df.set_index(["participant_id", "session_id"], inplace=True)

        for subject, session in train_df.index:
            if subject in test_df.index:
                flag_independant = False

        assert flag_independant


def check_subgroup_independence(train_path, test_path):
    print("Check subgroup independence")
    diagnosis_df_paths = os.listdir(test_path)
    diagnosis_df_paths = [x for x in diagnosis_df_paths if x.endswith("_baseline.tsv")]
    sub_diagnosis_list = [
        x for x in diagnosis_df_paths if "MCI" in x and x != "MCI_baseline.tsv"
    ]

    MCI_train_df = pd.read_csv(path.join(train_path, "MCI_baseline.tsv"), sep="\t")
    MCI_train_df.set_index(["participant_id", "session_id"], inplace=True)
    for sub_diagnosis in sub_diagnosis_list:
        flag_independant = True
        sub_test_df = pd.read_csv(path.join(test_path, sub_diagnosis), sep="\t")
        sub_test_df.set_index(["participant_id", "session_id"], inplace=True)

        for subject, session in MCI_train_df.index:
            if subject in sub_test_df.index:
                flag_independant = False

        assert flag_independant

    MCI_test_df = pd.read_csv(path.join(test_path, "MCI_baseline.tsv"), sep="\t")
    MCI_test_df.set_index(["participant_id", "session_id"], inplace=True)
    for sub_diagnosis in sub_diagnosis_list:
        flag_independant = True
        sub_test_df = pd.read_csv(path.join(train_path, sub_diagnosis), sep="\t")
        sub_test_df.set_index(["participant_id", "session_id"], inplace=True)

        for subject, session in MCI_test_df.index:
            if subject in sub_test_df.index:
                flag_independant = False

        assert flag_independant


def run_test_suite(formatted_data_path, n_splits, subset_name):
    check_train = True

    if n_splits == 0:
        train_path = path.join(formatted_data_path, "train")
        test_path = path.join(formatted_data_path, subset_name)
        if not path.exists(train_path):
            check_train = False

        check_subject_unicity(test_path)
        if check_train:
            check_subject_unicity(train_path)
            check_independance(train_path, test_path)
            MCI_path = path.join(train_path, "MCI_baseline.tsv")
            if path.exists(MCI_path):
                check_subgroup_independence(train_path, test_path)

    else:
        for split in range(n_splits):
            train_path = path.join(
                formatted_data_path,
                "train_splits-" + str(n_splits),
                "split-" + str(split),
            )
            test_path = path.join(
                formatted_data_path,
                subset_name + "_splits-" + str(n_splits),
                "split-" + str(split),
            )

            if not path.exists(train_path):
                check_train = False

            check_subject_unicity(test_path)
            if check_train:
                check_subject_unicity(train_path)
                check_independance(train_path, test_path)
                MCI_path = path.join(train_path, "MCI_baseline.tsv")
                if path.exists(MCI_path):
                    check_subgroup_independence(train_path, test_path)


def test_getlabels():
    """Checks that getlabels is working and that it is coherent with previous version in reference_path"""
    output_path = "data/tsvtool_test"
    flag_getlabels = not os.system(
        f"clinicadl -vvv tsvtool getlabels {merged_tsv} {missing_mods} {output_path} "
        f"--diagnoses {diagnoses}"
    )
    assert flag_getlabels
    for file in os.listdir(output_path):
        out_df = pd.read_csv(path.join(output_path, file), sep="\t")
        ref_df = pd.read_csv(path.join(reference_path, file), sep="\t")
        assert out_df.equals(ref_df)

    shutil.rmtree(output_path)


def test_split():
    """Checks that:
    -  split and kfold are working
    -  the loading functions can find the output
    -  no data leakage is introduced in split and kfold.
    """
    n_splits = 5
    train_path = path.join(reference_path, "train")
    flag_split = not os.system(f"clinicadl tsvtool split {reference_path} -vvv")
    flag_kfold = not os.system(
        f"clinicadl  -vvv tsvtool kfold {train_path} --n_splits {n_splits}"
    )
    assert flag_split
    assert flag_kfold
    flag_load = True
    try:
        _ = load_data_test(path.join(reference_path, "test"), diagnoses.split(" "))
        split_manager = KFoldSplit(".", train_path, diagnoses.split(" "), n_splits)
        for fold in split_manager.fold_iterator():
            _ = split_manager[fold]
    except FileNotFoundError:
        flag_load = False
    assert flag_load

    run_test_suite(reference_path, 0, "test")
    run_test_suite(path.join(reference_path, "train"), n_splits, "validation")

    shutil.rmtree(path.join(reference_path, "train"))
    shutil.rmtree(path.join(reference_path, "test"))


def test_analysis():
    """Checks that analysis can be performed"""
    results_path = path.join("data", "tsvtool", "analysis.tsv")
    ref_analysis_path = path.join("data", "tsvtool", "anonymous_analysis.tsv")
    flag_analysis = not os.system(
        f"clinicadl tsvtool analysis {merged_tsv} {reference_path} {results_path} "
        f"--diagnoses {diagnoses}"
    )
    assert flag_analysis
    ref_df = pd.read_csv(ref_analysis_path, sep="\t")
    out_df = pd.read_csv(results_path, sep="\t")
    assert out_df.equals(ref_df)
    os.remove(results_path)
