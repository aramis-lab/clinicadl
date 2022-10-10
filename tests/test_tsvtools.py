import os
import shutil
from os import path
from os.path import join
from pathlib import Path

import pandas as pd

from clinicadl.utils.caps_dataset.data import load_data_test
from clinicadl.utils.split_manager import KFoldSplit
from clinicadl.utils.tsvtools_utils import extract_baseline

# data_ci_directory = "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/"
# bids_directory = path.join(data_ci_directory, "ref/bids")
# in_directory = path.join(data_ci_directory, "in")
# reference_directory = path.join(data_ci_directory, "ref")
# output_directory = path.join(data_ci_directory, "out")

# labels_tsv = os.path.join(output_directory, "labels.tsv")
# merged_tsv = path.join(reference_directory, "merge-tsv.tsv")

# diagnoses = "CN pCN sCN usCN ukCN MCI sMCI pMCI usMCI ukMCI rMCI AD rAD sAD usAD ukAD"

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
            print(subject_df)
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


def run_test_suite(data_tsv, n_splits, subset_name):
    check_train = True

    if n_splits == 0:
        train_baseline_tsv = path.join(data_tsv, "train_baseline.tsv")
        test_baseline_tsv = path.join(data_tsv, "test_baseline.tsv")
        if not path.exists(train_baseline_tsv):
            check_train = False

        check_subject_unicity(test_baseline_tsv)
        if check_train:
            check_subject_unicity(train_baseline_tsv)
            check_independance(train_baseline_tsv, test_baseline_tsv)

    else:
        for split_number in range(n_splits):

            for folder, sub_folder, files in os.walk(path.join(data_tsv, "split")):
                for file in files:
                    if file[-3:] == "tsv":
                        check_subject_unicity(path.join(folder, file))
                train_baseline_tsv = path.join(folder, "train_baseline.tsv")
                test_baseline_tsv = path.join(folder, "test_baseline.tsv")
                if path.exists(train_baseline_tsv):
                    if path.exists(test_baseline_tsv):
                        check_independance(train_baseline_tsv, test_baseline_tsv)


def test_getlabels(cmdopt, tmp_path):
    # base_dir = Path(cmdopt["input"])
    # input_dir = base_dir / "train" / "in"
    # ref_dir = base_dir / "train" / "ref"
    # tmp_out_dir = tmp_path / "train" / "out"
    # tmp_out_dir.mkdir(parents=True)

    input_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/in"
    )
    ref_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/ref"
    )
    tmp_out_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/out"
    )

    """Checks that getlabels is working and that it is coherent with previous version in reference_path"""
    import shutil

    bids_output = path.join(tmp_out_dir, "bids")
    bids_directory = path.join(ref_dir, "bids")
    if path.exists(tmp_out_dir):
        shutil.rmtree(tmp_out_dir)
        os.makedirs(tmp_out_dir)
    shutil.copytree(bids_directory, bids_output)
    merged_tsv = path.join(ref_dir, "merge-tsv.tsv")
    missing_mods_directory = path.join(ref_dir, "missing_mods")

    flag_getlabels = not os.system(
        f"clinicadl -vvv tsvtools get-labels {bids_output} "
        f"-d AD -d MCI -d CN -d Dementia "
        f"--merged_tsv {merged_tsv} --missing_mods {missing_mods_directory}"
    )
    assert flag_getlabels

    out_df = pd.read_csv(path.join(tmp_out_dir, "labels.tsv"), sep="\t")
    ref_df = pd.read_csv(path.join(ref_dir, "labels.tsv"), sep="\t")
    assert out_df.equals(ref_df)

    # shutil.rmtree(output_directory)


def test_split(cmdopt, tmp_path):
    # base_dir = Path(cmdopt["input"])
    # input_dir = base_dir / "train" / "in"
    # ref_dir = base_dir / "train" / "ref"
    # tmp_out_dir = tmp_path / "train" / "out"
    # tmp_out_dir.mkdir(parents=True)

    input_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/in"
    )
    ref_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/ref"
    )
    tmp_out_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/tsvtools/out"
    )

    """Checks that:
    -  split and kfold are working
    -  the loading functions can find the output
    -  no data leakage is introduced in split and kfold.
    """
    n_splits = 3
    train_tsv = path.join(tmp_out_dir, "split/train.tsv")
    labels_tsv = path.join(tmp_out_dir, "labels.tsv")

    flag_split = not os.system(
        f"clinicadl -vvv tsvtools split {labels_tsv} --subset_name test"
    )
    flag_kfold = not os.system(
        f"clinicadl -vvv tsvtools kfold {train_tsv} --n_splits {n_splits} --subset_name validation"
    )
    assert flag_split
    assert flag_kfold

    run_test_suite(tmp_out_dir, n_splits, "validation")


def test_analysis(cmdopt, tmp_path):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "train" / "in"
    ref_dir = base_dir / "train" / "ref"
    tmp_out_dir = tmp_path / "train" / "out"
    tmp_out_dir.mkdir(parents=True)

    """Checks that analysis can be performed"""

    merged_tsv = path.join(ref_dir, "merge-tsv.tsv")
    labels_tsv = path.join(ref_dir, "labels.tsv")
    output_tsv = path.join(tmp_out_dir, "analysis.tsv")
    ref_analysis_tsv = path.join(ref_dir, "analysis.tsv")

    flag_analysis = not os.system(
        f"clinicadl tsvtools analysis {merged_tsv} {labels_tsv} {output_tsv} "
        f"--diagnoses CN --diagnoses MCI --diagnoses Dementia"
    )

    assert flag_analysis
    ref_df = pd.read_csv(ref_analysis_tsv, sep="\t")
    out_df = pd.read_csv(output_tsv, sep="\t")
    assert out_df.equals(ref_df)
