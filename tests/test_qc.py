import shutil
from os import system
from os.path import join
from pathlib import Path

import pandas as pd
import pytest

from tests.testing_tools import compare_folders


@pytest.fixture(params=["t1-linear", "t1-volume", "pet-linear"])
def test_name(request):
    return request.param


def test_qc(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "qualityCheck" / "in"
    ref_dir = base_dir / "qualityCheck" / "ref"
    tmp_out_dir = tmp_path / "qualityCheck" / "out"
    tmp_out_dir.mkdir(parents=True)

    if test_name == "t1-linear":
        out_tsv = str(tmp_out_dir / "QC.tsv")
        test_input = [
            "t1-linear",
            str(input_dir / "caps"),
            out_tsv,
            "--no-gpu",
        ]

    elif test_name == "t1-volume":
        out_dir = str(tmp_out_dir / "QC_T1V")
        test_input = [
            "t1-volume",
            str(input_dir / "caps_T1V"),
            out_dir,
            "Ixi549Space",
        ]

    elif test_name == "pet-linear":
        out_tsv = str(tmp_out_dir / "QC_pet.tsv")
        test_input = [
            "pet-linear",
            str(input_dir / "caps_pet"),
            out_tsv,
            "18FFDG",
            "cerebellumPons2",
            "--threshold",
            "0.5",
        ]
    else:
        raise NotImplementedError(
            f"Quality check test on {test_name} is not implemented, test."
        )

    flag_error = not system(f"clinicadl quality-check " + " ".join(test_input))
    assert flag_error

    if test_name == "t1-linear":

        ref_tsv = join(ref_dir, "QC.tsv")
        ref_df = pd.read_csv(ref_tsv, sep="\t")
        ref_df.reset_index(inplace=True)

        out_df = pd.read_csv(out_tsv, sep="\t")
        out_df.reset_index(inplace=True)

        out_df["pass_probability"] = round(out_df["pass_probability"], 2)
        ref_df["pass_probability"] = round(ref_df["pass_probability"], 2)

        system(f"diff {out_tsv} {ref_tsv} ")
        assert out_df.equals(ref_df)

    elif test_name == "t1-volume":
        assert compare_folders(out_dir, str(ref_dir / "QC_T1V"), tmp_out_dir)

    elif test_name == "pet-linear":
        out_df = pd.read_csv(out_tsv, sep="\t")
        ref_tsv = join(ref_dir, "QC_pet.tsv")
        ref_df = pd.read_csv(ref_tsv, sep="\t")
        out_df.reset_index(inplace=True)
        ref_df.reset_index(inplace=True)
        out_df["pass_probability"] = round(out_df["pass_probability"], 2)
        ref_df["pass_probability"] = round(ref_df["pass_probability"], 2)
        system(f"diff {out_tsv} {ref_tsv} ")
        assert out_df.equals(ref_df)
