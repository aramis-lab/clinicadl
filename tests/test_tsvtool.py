import os
import shutil
from os import path

import pandas as pd

from clinicadl.utils.caps_dataset.data import load_data_test
from clinicadl.utils.split_manager import KFoldSplit
from clinicadl.utils.tsvtools_utils import extract_baseline

# bids_tar = "data/tsvtool/OasisBids.tar.gz"
merged_tsv = "data/tsvtool/anonymous_BIDS.tsv"
missing_mods = "data/tsvtool/anonymous_missing_mods"
reference_path = "data/tsvtool/anonymous_reference"
diagnoses = "CN pCN sCN usCN ukCN MCI sMCI pMCI usMCI ukMCI rMCI AD rAD sAD usAD ukAD"

"""
Check the absence of data leakage
    1) Baseline datasets contain only one scan per subject
    2) No intersection between train and test sets
    3) Absence of MCI train subjects in test sets of subcategories of MCI
"""


def check_subject_unicity(labels_path_baseline):
    print("Check unicity", labels_path_baseline)

    flag_unique = True
    check_df = pd.read_csv(labels_path_baseline, sep="\t")
    check_df.set_index(["participant_id", "session_id"], inplace=True)
    if labels_path_baseline[-12:] != "baseline.tsv":
        check_df = extract_baseline(check_df, set_index=False)
    for subject, subject_df in check_df.groupby(level=0):
        if len(subject_df) > 1:
            flag_unique = False
    assert flag_unique


def check_independance(train_path_baseline, test_path_baseline, subject_flag=True):
    print("Check independence")

    flag_independant = True
    train_df = pd.read_csv(train_path_baseline, sep="\t")
    train_df.set_index(["participant_id", "session_id"], inplace=True)
    test_df = pd.read_csv(test_path_baseline, sep="\t")
    test_df.set_index(["participant_id", "session_id"], inplace=True)

    for subject, session in train_df.index:
        if subject_flag:
            if subject in test_df.index:
                flag_independant = False
        else:
            if (subject, session) in test_df.index:
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
        train_path = path.join(formatted_data_path, "validation.tsv")
        test_path_baseline = path.join(formatted_data_path, "test_baseline.tsv")
        if not path.exists(train_path):
            check_train = False

        check_subject_unicity(test_path_baseline)
        if check_train:
            check_subject_unicity(train_path)
            check_independance(train_path, test_path_baseline)

    else:
        for split_number in range(n_splits):
            train_path = path.join(formatted_data_path, "validation.tsv")
            test_path_baseline = path.join(formatted_data_path, "test_baseline.tsv")

            if not path.exists(train_path):
                check_train = False

            check_subject_unicity(test_path_baseline)

            train_df = pd.read_csv(train_path, sep="\t")
            split_df = train_df.loc[train_df["split"] == split_number]
            train_split_df = split_df.loc[split_df["datagroup"] == "train"]
            train_split_path = path.join(formatted_data_path, "train_split.tsv")
            train_split_df.to_csv(train_split_path, sep="\t", index=False)

            subset_df = split_df.loc[split_df["datagroup"] == subset_name]
            subset_path = path.join(formatted_data_path, "subset.tsv")
            subset_df.to_csv(subset_path, sep="\t", index=False)
            if check_train:
                check_subject_unicity(train_split_path)
                check_subject_unicity(subset_path)
                check_independance(
                    train_split_path, test_path_baseline, subject_flag=True
                )
                check_independance(train_split_path, subset_path, subject_flag=False)

        os.remove(subset_path)
        os.remove(train_split_path)


def test_getlabels():
    """Checks that getlabels is working and that it is coherent with previous version in reference_path"""
    output_path = "data/tsvtools_test"
    bids_directory = "anonymous_bids/OasisBids_example"

    flag_getlabels = not os.system(
        f"clinicadl -vvv tsvtools getlabels {bids_directory} {output_path} "
        f"-d AD -d MCI -d CN "
        f"--merge_tsv {merged_tsv} --missing_mods {missing_mods}"
    )
    assert flag_getlabels

    file = "getlabels.tsv"
    out_df = pd.read_csv(path.join(output_path, file), sep="\t")
    out_df = out_df.sort_values(by=["participant_id", "session_id"])
    out_df.drop(columns=["subgroup"])
    out_df = out_df[["participant_id", "session_id", "age", "sex", "group"]]
    ref_df = pd.read_csv(path.join(reference_path, file), sep="\t")
    assert out_df.equals(ref_df)

    flag_getlabels2 = not os.system(
        f"clinicadl -vvv tsvtools getlabels {bids_directory} {output_path} "
        f"--diagnoses rAD --diagnoses usCN --diagnoses pMCI "
        f"--merge_tsv {merged_tsv} --missing_mods {missing_mods}"
    )
    assert flag_getlabels2

    file2 = "getlabels2.tsv"
    out_df = pd.read_csv(path.join(output_path, file), sep="\t")
    out_df = out_df.sort_values(by=["participant_id", "session_id"])
    out_df.drop(columns=["subgroup"])
    out_df = out_df[["participant_id", "session_id", "age", "sex", "group"]]
    ref_df = pd.read_csv(path.join(reference_path, file2), sep="\t")
    assert out_df.equals(ref_df)

    shutil.rmtree(output_path)


def test_split():
    """Checks that:
    -  split and kfold are working
    -  the loading functions can find the output
    -  no data leakage is introduced in split and kfold.
    """
    n_splits = 5
    getlabels_path = path.join(reference_path, "labels.tsv")
    test_path = path.join(reference_path, "test_baseline.tsv")
    flag_split = not os.system(
        f"clinicadl -vvv tsvtools split {getlabels_path} --subset_name test"
    )
    flag_kfold = not os.system(
        f"clinicadl -vvv tsvtools kfold {getlabels_path} --n_splits {n_splits} --test_tsv {test_path}"
    )
    assert flag_split
    assert flag_kfold
    # flag_load = True
    # try:
    #     _ = load_data_test(
    #         path.join(reference_path, "validation"), diagnoses.split(" ")
    #     )
    #     split_manager = KFoldSplit(".", reference_path, diagnoses.split(" "), n_splits)
    #     for split in split_manager.split_iterator():
    #         _ = split_manager[split]
    # except FileNotFoundError:
    #     flag_load = False
    # assert flag_load

    run_test_suite(reference_path, n_splits, "validation")
    # run_test_suite(reference_path, 0, "validation")

    # os.remove(test_path)

    # shutil.rmtree(path.join(reference_path, "train"))
    # shutil.rmtree(path.join(reference_path, "validation"))
    # shutil.rmtree(path.join(reference_path, "train_splits-5"))
    # shutil.rmtree(path.join(reference_path, "validation_splits-5"))


def test_analysis():
    """Checks that analysis can be performed"""
    results_path = path.join("data", "tsvtool", "analysis.tsv")
    ref_analysis_path = path.join("data", "tsvtool", "anonymous_analysis.tsv")
    getlabels_path = path.join(reference_path, "getlabels.tsv")

    flag_analysis = not os.system(
        f"clinicadl tsvtools analysis {merged_tsv} {getlabels_path} {results_path} "
        f"--diagnoses AD --diagnoses CN --diagnoses MCI"
    )

    assert flag_analysis
    ref_df = pd.read_csv(ref_analysis_path, sep="\t")
    out_df = pd.read_csv(results_path, sep="\t")
    out_df_sorted = out_df.reindex(sorted(out_df.columns), axis=1)
    ref_df_sorted = ref_df.reindex(sorted(ref_df.columns), axis=1)
    assert out_df_sorted.equals(ref_df_sorted)
    os.remove(results_path)
