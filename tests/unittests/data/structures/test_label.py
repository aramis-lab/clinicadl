from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from clinicadl.data.structures import Column, Mask


def test_Column():
    c = Column("age")
    assert c == "age"
    assert str(c) == "Column('age')"


def test_Mask():
    caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
    subject_dir = caps_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear"

    #########################################################
    mask = Mask(str(caps_dir / "masks" / "leftHippocampus.nii.gz"))
    assert mask.path == caps_dir / "masks" / "leftHippocampus.nii.gz"
    assert mask.name == "leftHippocampus"
    assert str(mask) == f"Mask('{str(mask.path)}')"

    mask = Mask(caps_dir / "masks" / "leftHippocampus.nii.gz")
    assert mask.is_common_mask
    assert mask._mask_img is None
    associated_mask = mask.get_associated_mask()
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(caps_dir / "masks" / "leftHippocampus.nii.gz").get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.affine, np.diag([1.3, 1.2, 1.1, 1])).all()
    assert mask._mask_img is not None

    # tensor mask
    mask = Mask(caps_dir / "masks" / "tensors" / "leftHippocampus.pt")
    associated_mask = mask.get_associated_mask()
    assert (
        associated_mask.tensor
        == torch.load(
            caps_dir / "masks" / "tensors" / "leftHippocampus.pt", weights_only=True
        )["mask"]
    ).all()

    #####################################
    mask = Mask("brain")
    assert not mask.is_common_mask
    assert str(mask) == "Mask('brain')"

    associated_mask = mask.get_associated_mask(
        subject_dir / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(
                subject_dir
                / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_brain.nii.gz"
            ).get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.affine, np.diag([1.3, 1.2, 1.1, 1])).all()
    assert mask._mask_img is None

    ##### errors #####
    with pytest.raises(FileNotFoundError):
        mask = Mask(Path("abc.nii.gz"))

    mask = Mask("brain")
    with pytest.raises(FileNotFoundError):
        mask.get_associated_mask(
            subject_dir.parent
            / "pet_linear"
            / "tensors"
            / "sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii"
        )
    with pytest.raises(ValueError):
        mask.get_associated_mask()
