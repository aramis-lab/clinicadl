import os
import shutil
import pandas as pd
from os import path
from clinicadl.tools.deep_learning.data import load_data, load_data_test
from clinicadl.tools.tsv.test import run_test_suite

merged_tsv = "data/ADNI_BIDS.tsv"
missing_mods = "data/ADNI_missing_mods"
reference_path = "data/tsvtool_ref"
diagnoses = "AD CN MCI pMCI sMCI"


def test_getlabels():
    """Checks that getlabels is working and that it is coherent with previous version in reference_path"""
    output_path = "data/tsvtool_test"
    flag_getlabels = not os.system("clinicadl tsvtool getlabels %s %s %s --diagnoses %s"
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
    flag_split = not os.system("clinicadl tsvtool split %s %s --age_name age"
                               % (merged_tsv, reference_path))
    flag_kfold = not os.system("clinicadl tsvtool kfold %s --n_splits %i"
                               % (path.join(reference_path, "train"), n_splits))
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
    results_path = "data/analysis.tsv"
    flag_analysis = not os.system("clinicadl tsvtool analysis %s %s %s --age_name age --mmse_name MMSE"
                                  % (merged_tsv, reference_path, results_path))
    assert flag_analysis
    os.remove(results_path)
