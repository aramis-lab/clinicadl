import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader.batch import Batch
from clinicadl.data.structures import DataPoint, Sample, Sample2D
from clinicadl.io.bids import BidsFileType

FILE_TYPE = BidsFileType(data_type="anat", suffix="T1w")


def test_init():
    with pytest.raises(ValueError, match="The batch is empty!"):
        Batch([])


def test_typing():
    datapoint = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
        participant="abc",
        session="abc",
    )
    batch = Batch([datapoint, datapoint])
    assert batch[0].participant == "abc"

    datapoint = Sample(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
        participant="abc",
        session="abc",
        image_path="abc.nii.gz",
        file_type=FILE_TYPE,
    )
    batch = Batch([datapoint, datapoint])
    assert str(batch[0].image_path[0]) == "abc.nii.gz"


def test_get_field():
    # torchio images
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            mask=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = Batch(list_samples)
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 4, 5)
    masks = batch.get_field("mask")
    assert masks.size() == (2, 1, 3, 4, 5)

    # tensors and different shapes
    batch[1] = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
        mask=tio.LabelMap(tensor=torch.ones(1, 3, 4, 6)),
        participant="sub-1",
        session="ses-1",
    )
    masks = batch.get_field("mask")
    images = batch.get_field("image")
    assert isinstance(images, list)
    assert images[0].size() == torch.Size((1, 3, 4, 5))
    assert images[1].size() == torch.Size((1, 3, 4, 6))
    assert isinstance(masks, list)
    assert masks[0].size() == torch.Size((1, 3, 4, 5))

    # numpy and list
    batch[0]["x"] = np.ones((1, 3, 4, 5)).tolist()
    batch[1]["x"] = np.ones((1, 3, 4, 5))
    assert batch.get_field("x").size() == (2, 1, 3, 4, 5)

    # None
    batch[1]["x"] = None
    x = batch.get_field("x")
    assert isinstance(x, list)
    assert x[0].size() == torch.Size((1, 3, 4, 5))
    assert x[1] is None

    # inhomogeneous numerics
    batch[0]["x"] = [0, 1, 2]
    batch[1]["x"] = [0, 1]
    x = batch.get_field("x")
    assert isinstance(x, list)
    assert x[0].size() == torch.Size((3,))
    assert x[1].size() == torch.Size((2,))

    # homogeneous numerics
    batch[0]["x"] = 0
    batch[1]["x"] = 1
    x = batch.get_field("x")
    torch.testing.assert_close(x, torch.tensor([0, 1], dtype=torch.int64))

    batch[0]["x"] = 0.0
    batch[1]["x"] = 1.0
    x = batch.get_field("x")
    torch.testing.assert_close(x, torch.tensor([0.0, 1.0], dtype=torch.float32))

    # dtype
    x = batch.get_field("x", dtype=torch.int64)
    torch.testing.assert_close(x, torch.tensor([0.0, 1.0], dtype=torch.int64))

    # channels
    batch[0]["x"] = 0
    batch[1]["x"] = 1
    x = batch.get_field("x", ensure_channel_dim=True)
    torch.testing.assert_close(x, torch.tensor([[0], [1]]))

    # slices
    batch = Batch(
        [
            Sample2D(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 1, 5)),
                mask=tio.LabelMap(tensor=torch.ones(1, 3, 1, 5)),
                participant=f"sub-{i}",
                session=f"ses-{i}",
                image_path="abc.nii.gz",
                file_type=FILE_TYPE,
                squeeze=False,
                slice_direction=1,
                sample_position=0,
            )
            for i in range(2)
        ]
    )
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 1, 5)

    batch[0].squeeze = batch[1].squeeze = True
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 5)
    masks = batch.get_field("mask")
    assert masks.size() == (2, 1, 3, 5)
    assert batch[0].image.tensor.shape == (1, 3, 1, 5)

    # errors
    with pytest.raises(
        KeyError,
        match="You want to get 'abc', but there is no such key in some DataPoints in the batch.",
    ):
        batch.get_field("abc")


@pytest.mark.gpu
def test_to():
    # device
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
                mask=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3)),
                output=1,
                abc=torch.randn(1, 3, 3, device=torch.device("cuda:0")),
                participant=f"sub-{i}",
                session=f"ses-{i}",
            )
            for i in range(2)
        ]
    )

    assert not batch._non_blocking
    assert batch.device is None
    assert not batch.channels_last
    assert batch[0].mask.tensor.device == torch.device("cpu")
    assert batch[0]["abc"].device == torch.device("cuda:0")
    assert batch.get_field("output").device == torch.device("cpu")
    assert batch.get_field("mask").stride() == (27, 27, 9, 3, 1)
    assert batch.get_field("abc").stride() == (9, 9, 3, 1)

    with pytest.raises(
        ValueError,
        match="If 'device' is a str, it must be 'cpu', 'cuda' or 'cuda:<device-id>'.",
    ):
        batch.to("cuda-0", non_blocking=True)
    batch.to("cuda", non_blocking=True)
    batch.to(torch.device("cuda:0"), non_blocking=True)

    batch.to(0, non_blocking=True, channels_last=True)

    assert batch._non_blocking
    assert batch.device == torch.device("cuda:0")
    assert batch.channels_last
    assert batch[0].mask.tensor.device == torch.device("cpu")
    assert batch[0]["abc"].device == torch.device("cuda:0")
    assert batch.get_field("output").device == torch.device("cuda:0")
    assert batch.get_field("mask").stride() == (27, 1, 9, 3, 1)
    assert batch.get_field("abc").stride() == (9, 1, 3, 1)

    batch.to("cpu")

    assert not batch._non_blocking
    assert batch.device == torch.device("cpu")
    assert batch.channels_last
    assert batch.get_field("output").device == torch.device("cpu")
    assert batch.get_field("mask").stride() == (27, 1, 9, 3, 1)
    assert batch.get_field("abc").stride() == (9, 1, 3, 1)

    batch.to(channels_last=False)
    assert not batch.channels_last
    assert batch.get_field("mask").stride() == (27, 27, 9, 3, 1)


def test_add_field():
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
                participant=f"sub-{i}",
                session=f"ses-{i}",
            )
            for i in range(2)
        ]
    )
    batch.add_field(torch.tensor([1, 2]), "abc")
    torch.testing.assert_close(batch[0]["abc"], torch.tensor(1))
    torch.testing.assert_close(batch[1]["abc"], torch.tensor(2))
    with pytest.raises(
        AssertionError,
        match="'values' must have the same length as the batch. Got 3 values, whereas the batch has only 2 elements",
    ):
        batch.add_field("bcd", [1, 2, 3])

    batch.add_images(torch.randn(2, 1, 10, 10, 10), "new_image")
    batch.add_masks(torch.randn(2, 1, 10, 10, 10), "new_mask")
    assert isinstance(batch[0]["new_image"], tio.ScalarImage)
    assert isinstance(batch[1]["new_image"], tio.ScalarImage)
    assert isinstance(batch[0]["new_mask"], tio.LabelMap)
    assert isinstance(batch[1]["new_mask"], tio.LabelMap)
