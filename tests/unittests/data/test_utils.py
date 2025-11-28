import shutil
from pathlib import Path

from clinicadl.data.utils import remove_tensors

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"


def test_remove_tensors(tmp_path):
    shutil.copytree(caps_dir, tmp_path, dirs_exist_ok=True)

    assert (
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_path / "masks" / "tensors" / "t1_masks" / "leftHippocampus.pt"
    ).is_file()
    assert (tmp_path / "tensor_conversion" / "t1_masks.json").is_file()

    remove_tensors(tmp_path / "tensor_conversion" / "t1_masks.json")
    assert not (tmp_path / "tensor_conversion" / "t1_masks.json").is_file()
    assert not (
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert not (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert not (
        tmp_path / "masks" / "tensors" / "t1_masks" / "leftHippocampus.pt"
    ).is_file()
