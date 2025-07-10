import shutil
from pathlib import Path

import pytest
import torch

from clinicadl.data.utils import remove_tensors

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
tmp_dir = Path(__file__).parents[1] / "resources" / "caps_tmp"


def copy_caps():
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)


@pytest.mark.parametrize(
    "conversion_name,tensor_dir",
    [("default_t1-linear", "default"), ("t1_masks", "t1_masks")],
)
def test_remove_tensors(conversion_name, tensor_dir):
    copy_caps()
    sub_ses_dir = (
        tmp_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear" / "tensors"
    )
    sub_ses_dir_bis = (
        tmp_dir / "subjects" / "sub-010" / "ses-M003" / "t1_linear" / "tensors"
    )
    torch.save(dict(), sub_ses_dir_bis / tensor_dir / "abc.pt")

    remove_tensors(str(tmp_dir), conversion_name)

    assert (
        not (tmp_dir / "tensor_conversion" / conversion_name)
        .with_suffix(".json")
        .exists()
    )
    assert not (sub_ses_dir / tensor_dir).exists()
    assert not (
        sub_ses_dir_bis
        / tensor_dir
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).exists()
    assert (sub_ses_dir_bis / tensor_dir / "abc.pt").exists()

    shutil.rmtree(tmp_dir)
