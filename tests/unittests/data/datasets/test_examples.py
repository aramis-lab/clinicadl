import pytest

from clinicadl.data.datasets.examples import BidsStroke, BidsStrokeSmall
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
