from pathlib import Path

import torchio as tio

from clinicadl.data.structures import CommonMask, Image, IndividualMask
from clinicadl.io import Bids, BidsFileType, T1Linear

CAPS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "caps"
MASKS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "masks"


def test_image():
    caps = Bids(CAPS)
    file_type = T1Linear(use_uncropped_image=True)
    image = Image(caps, file_type)
    assert isinstance(image.get("sub-000", "ses-M000"), tio.ScalarImage)


def test_individual_mask():
    masks = Bids(MASKS)
    file_type = BidsFileType(data_type="anat", suffix="dseg")
    mask = IndividualMask(masks, file_type)
    assert isinstance(mask.get("sub-000", "ses-M000"), tio.LabelMap)


def test_common_mask():
    mask = CommonMask(
        CAPS / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
    )
    assert isinstance(mask.get(), tio.LabelMap)
    assert mask._mask is not None
    assert isinstance(mask.get(), tio.LabelMap)
