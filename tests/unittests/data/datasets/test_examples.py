from unittest.mock import patch

import pytest

from clinicadl.data.datasets.examples import (
    BidsDLBS,
    BidsDLBSSmall,
    BidsNeuroEmo,
    BidsStroke,
    BidsStrokeSmall,
    CapsDLBS,
)
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import CropOrPadConfig


@pytest.mark.parametrize("bids,len_", [(BidsStrokeSmall, 10), (BidsStroke, 50)])
def test_bids_stroke(bids, len_):
    dataset = bids(
        transforms=TransformsHandler(
            image_transforms=[CropOrPadConfig(target_shape=16)]
        ),
        masks=True,
        columns=["age"],
    )
    assert len(dataset) == len_
    sample = dataset[0]
    assert "age" in sample
    assert sample.image.spatial_shape == (16, 16, 16)
    assert sample["lesion_mask"].spatial_shape == (16, 16, 16)

    dataset = bids(
        masks=False,
    )
    assert "lesion_mask" not in dataset[0]


@pytest.mark.parametrize(
    "task,n_time_points",
    [
        (
            "fe",
            200,
        ),
        (
            "rest",
            250,
        ),
    ],
)
def test_bids_neuro_emo(task, n_time_points):
    dataset = BidsNeuroEmo(
        task=task,
        transforms=TransformsHandler(
            image_transforms=[CropOrPadConfig(target_shape=16)]
        ),
    )
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample.image.shape == (n_time_points, 16, 16, 16)


@pytest.mark.parametrize(
    "bids,pet,column,len_",
    [
        (
            BidsDLBS,
            False,
            "AgeMRI",
            66,
        ),
        (
            BidsDLBS,
            True,
            "AgePETAmy",
            61,
        ),
        (
            BidsDLBSSmall,
            False,
            "AgeMRI",
            30,
        ),
        (
            BidsDLBSSmall,
            True,
            "AgePETAmy",
            24,
        ),
    ],
)
def test_bids_dlbs(bids, pet, column, len_):
    dataset = bids(
        pet=pet,
        transforms=TransformsHandler(
            image_transforms=[CropOrPadConfig(target_shape=16)]
        ),
        columns=[column],
    )

    assert len(dataset) == len_
    sample = dataset[0]
    assert column in sample
    assert sample.image.spatial_shape == (16, 16, 16)


@pytest.mark.parametrize("bids", [BidsDLBS, BidsDLBSSmall])
def test_bids_dlbs_download_pet_after(bids: type[BidsDLBS], tmp_path):
    with patch(
        "clinicadl.data.datasets.examples.get_clinicadl_cache_dir"
    ) as mocked_func:
        mocked_func.return_value = tmp_path

        suffix = "Small" if "Small" in type(bids).__name__ else ""
        bids_multimodal = bids(pet=False)
        assert not bids_multimodal.dir.name.endswith("Pet" + suffix)
        assert not bids_multimodal.download_url.endswith("Pet" + suffix)

        bids_pet = bids(pet=True)
        assert not bids_pet.dir.name.endswith("Pet" + suffix)
        assert not bids_pet.dir.name.endswith("Pet" + suffix)


@pytest.mark.parametrize("bids", [BidsDLBS, BidsDLBSSmall])
def test_bids_dlbs_download_pet_before(bids: type[BidsDLBS], tmp_path):
    with patch(
        "clinicadl.data.datasets.examples.get_clinicadl_cache_dir"
    ) as mocked_func:
        mocked_func.return_value = tmp_path

        suffix = "Small" if "Small" in type(bids).__name__ else ""

        bids_pet = bids(pet=True)
        assert bids_pet.dir.name.endswith("Pet" + suffix)
        assert bids_pet.dir.name.endswith("Pet" + suffix)


@pytest.mark.parametrize(
    "pet,cropped,column,transforms,expected_shape",
    [
        (
            False,
            False,
            "AgeMRI",
            [CropOrPadConfig(target_shape=16)],
            (16, 16, 16),
        ),
        (
            True,
            False,
            "AgePETAmy",
            [CropOrPadConfig(target_shape=16)],
            (16, 16, 16),
        ),
        (
            False,
            True,
            "AgeMRI",
            [],
            (169, 208, 179),
        ),
        (
            False,
            False,
            "AgeMRI",
            [],
            (193, 229, 193),
        ),
    ],
)
def test_caps_dlbs(pet, cropped, column, transforms, expected_shape):
    dataset = CapsDLBS(
        pet=pet,
        cropped=cropped,
        transforms=TransformsHandler(
            image_transforms=transforms,
        ),
        columns=[column],
    )

    assert len(dataset) == 5
    sample = dataset[0]
    assert column in sample
    assert sample.image.spatial_shape == expected_shape
