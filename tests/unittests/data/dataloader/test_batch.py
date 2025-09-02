import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader.batch import Batch, simple_collate_fn, tuple_collate_fn
from clinicadl.data.structures import DataPoint


def test_init():
    with pytest.raises(ValueError, match="The batch is empty!"):
        Batch([])


def test_get_field():
    # torchio images
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = Batch(list_samples)
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 4, 5)
    labels = batch.get_field("label")
    assert labels.size() == (2, 1, 3, 4, 5)

    # tensors and different shapes
    batch[-1] = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
        label=torch.ones(1, 3, 4, 6),
        participant="sub-1",
        session="ses-1",
    )
    labels = batch.get_field("label")
    images = batch.get_field("image")
    assert isinstance(images, list)
    assert isinstance(images[0], tio.ScalarImage)
    assert isinstance(images[-1], tio.ScalarImage)
    assert isinstance(labels, list)
    assert isinstance(labels[0], tio.LabelMap)
    assert labels[-1].size() == (1, 3, 4, 6)

    # numpy and list
    batch[0]["label"] = list(np.ones((1, 3, 4, 5)))
    batch[-1]["label"] = np.ones((1, 3, 4, 5))
    labels = batch.get_field("label")
    assert labels.size() == (2, 1, 3, 4, 5)

    # None
    batch[-1]["label"] = None
    labels = batch.get_field("label")
    assert isinstance(labels, list)
    assert isinstance(labels[0], list)
    assert labels[-1] is None

    # dict
    batch[0]["label"] = {"A": 0.0, "B": 1.0}
    batch[-1]["label"] = {"A": 2.0, "B": 3.0}
    labels = batch.get_field("label")
    torch.testing.assert_close(
        labels, torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    )

    batch[0]["label"] = {"A": 0, "B": "abc"}
    labels = batch.get_field("label")
    assert isinstance(labels, list)
    assert isinstance(labels[0], dict)

    # homogeneous numerics
    batch[0]["label"] = 0
    batch[-1]["label"] = 1
    labels = batch.get_field("label")
    torch.testing.assert_close(labels, torch.tensor([0, 1], dtype=torch.int64))

    batch[0]["label"] = 0.0
    batch[-1]["label"] = 1.0
    labels = batch.get_field("label")
    torch.testing.assert_close(labels, torch.tensor([0.0, 1.0], dtype=torch.float32))

    # dtype
    labels = batch.get_field("label", dtype=torch.int64)
    torch.testing.assert_close(labels, torch.tensor([0.0, 1.0], dtype=torch.int64))

    # channels
    batch[0]["label"] = torch.ones(1, 3, 4, 5)
    batch[1]["label"] = torch.ones(1, 3, 4, 5)
    labels = batch.get_field("label", channels_last=True)
    assert labels.stride() == (60, 1, 20, 5, 1)
    labels = batch.get_field("label", channels_last=False)
    assert labels.stride() == (60, 60, 20, 5, 1)

    batch[0]["label"] = torch.ones(1, 3, 4)
    batch[1]["label"] = torch.ones(1, 3, 4)
    labels = batch.get_field("label", channels_last=True)
    assert labels.stride() == (12, 1, 4, 1)
    labels = batch.get_field("label", channels_last=False)
    assert labels.stride() == (12, 12, 4, 1)

    batch[0]["label"] = 0
    batch[1]["label"] = 1
    labels = batch.get_field("label", ensure_channel_dim=True)
    torch.testing.assert_close(labels, torch.tensor([[0], [1]]))

    # errors
    with pytest.raises(
        ValueError,
        match=r"To use Channels Last memory format, the output tensor must be 4D \(BCHW\) or 5D \(BCDHW\). Here it is 1D.",
    ):
        batch.get_field("label", channels_last=True)
    with pytest.raises(
        KeyError,
        match="You want to get 'abc', but there is no such key in some DataPoints in the batch.",
    ):
        batch.get_field("abc")


@pytest.mark.gpu
def test_to():
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
                label=torch.randn(2),
                output=1,
                abc=torch.randn(2, device=torch.device("cuda:0")),
                participant=f"sub-{i}",
                session=f"ses-{i}",
            )
            for i in range(2)
        ]
    )
    batch_gpu = batch.to(0, non_blocking=True)
    assert batch_gpu._non_blocking
    assert batch_gpu.device == torch.device("cuda:0")
    assert not batch._non_blocking
    assert batch.device is None

    assert batch[0]["label"].device == torch.device("cpu")
    assert batch_gpu[0]["label"].device == torch.device("cuda:0")

    assert batch[0]["abc"].device == torch.device("cuda:0")
    assert batch_gpu[0]["abc"].device == torch.device("cuda:0")

    assert batch.get_field("output").device == torch.device("cpu")
    assert batch_gpu.get_field("output").device == torch.device("cuda:0")


def test_simple_collate_fn():
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    out = simple_collate_fn(list_samples)
    assert isinstance(out, Batch)


def test_tuple_collate_fn():
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    out = tuple_collate_fn(list(zip(list_samples, list_samples)))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], Batch)
