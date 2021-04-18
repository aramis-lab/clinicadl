import os
import shutil
import pandas as pd
from os import path
from clinicaaddl.tools.deep_learning.data import load_data, load_data_test
from clinicaaddl.tools.tsv.test import run_test_suite

merged_tsv = "data/tsvtool/anonymous_BIDS.tsv"
missing_mods = "data/tsvtool/anonymous_missing_mods"
reference_path = "data/tsvtool/anonymous_reference"
diagnoses = "AD CN MCI pMCI sMCI"


def test_getlabels():
    """Checks that getlabels is working and that it is coherent with previous version in reference_path"""
    output_path = "data/tsvtool_test"
    flag_getlabels = not os.system("clinicaaddl tsvtool getlabels %s %s %s --diagnoses %s -vvv"
                                   % (merged_tsv, missing_mods, output_path, diagnoses))
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
    flag_split = not os.system(f"clinicaaddl tsvtool split {reference_path} -vvv")
    flag_kfold = not os.system(f"clinicaaddl tsvtool kfold {train_path} --n_splits {n_splits} -vvv")
    assert flag_split
    assert flag_kfold
    flag_load = True
    try:
        _ = load_data_test(path.join(reference_path, "test"), diagnoses.split(" "))
        for fold in range(n_splits):
            _, _ = load_data(path.join(reference_path, "train"), diagnoses.split(" "), fold, n_splits=n_splits)
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
    flag_analysis = not os.system("clinicaaddl tsvtool analysis %s %s %s "
                                  "--diagnoses AD CN MCI sMCI pMCI"
                                  % (merged_tsv, reference_path, results_path))
    assert flag_analysis
    ref_df = pd.read_csv(ref_analysis_path, sep="\t")
    out_df = pd.read_csv(results_path, sep="\t")
    assert out_df.equals(ref_df)
    os.remove(results_path)
