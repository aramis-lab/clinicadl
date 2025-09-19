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
    batch[0]["label"] = np.ones((1, 3, 4, 5)).tolist()
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
    batch[0]["label"] = 0
    batch[1]["label"] = 1
    labels = batch.get_field("label", ensure_channel_dim=True)
    torch.testing.assert_close(labels, torch.tensor([[0], [1]]))

    # slices
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 1, 5)),
                label=tio.LabelMap(tensor=torch.ones(1, 3, 1, 5)),
                participant=f"sub-{i}",
                session=f"ses-{i}",
                squeeze=False,
                slice_direction=1,
            )
            for i in range(2)
        ]
    )
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 1, 5)

    batch[0].squeeze = batch[1].squeeze = True
    images = batch.get_field("image")
    assert images.size() == (2, 1, 3, 5)
    labels = batch.get_field("label")
    assert labels.size() == (2, 1, 3, 5)
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
                label=torch.randn(1, 3, 3, 3),
                output=1,
                abc=torch.randn(1, 3, 3, device=torch.device("cuda:0")),
                participant=f"sub-{i}",
                session=f"ses-{i}",
            )
            for i in range(2)
        ]
    )

    # no changes
    batch_bis = batch.to()
    assert batch_bis is batch

    # changes
    with pytest.raises(
        ValueError,
        match="If 'device' is a str, it must be 'cpu', 'cuda' or 'cuda:<device-id>'.",
    ):
        batch.to("cuda-0", non_blocking=True)
    batch.to("cuda", non_blocking=True)
    batch.to(torch.device("cuda:0"), non_blocking=True)

    batch_gpu = batch.to(0, non_blocking=True, channels_last=True)
    batch_cpu = batch.to("cpu")

    assert batch_gpu._non_blocking
    assert batch_gpu.device == torch.device("cuda:0")
    assert batch_gpu.channels_last
    assert batch_gpu[0]["label"].device == torch.device("cuda:0")
    assert batch_gpu[0]["abc"].device == torch.device("cuda:0")
    assert batch_gpu.get_field("output").device == torch.device("cuda:0")
    assert batch_gpu.get_field("label").stride() == (27, 1, 9, 3, 1)
    assert batch_gpu.get_field("abc").stride() == (9, 1, 3, 1)

    assert not batch_cpu._non_blocking
    assert batch_cpu.device == torch.device("cpu")
    assert not batch_cpu.channels_last
    assert batch_cpu[0]["label"].device == torch.device("cpu")
    assert batch_cpu[0]["abc"].device == torch.device("cpu")
    assert batch_cpu.get_field("output").device == torch.device("cpu")
    assert batch_cpu.get_field("label").stride() == (27, 27, 9, 3, 1)
    assert batch_cpu.get_field("abc").stride() == (9, 9, 3, 1)

    assert not batch._non_blocking
    assert batch.device is None
    assert not batch.channels_last
    assert batch[0]["label"].device == torch.device("cpu")
    assert batch[0]["abc"].device == torch.device("cuda:0")
    assert batch.get_field("output").device == torch.device("cpu")
    assert batch.get_field("label").stride() == (27, 27, 9, 3, 1)
    assert batch.get_field("abc").stride() == (9, 9, 3, 1)


def test_add_field():
    batch = Batch(
        [
            DataPoint(
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
                label=None,
                participant=f"sub-{i}",
                session=f"ses-{i}",
            )
            for i in range(2)
        ]
    )
    batch.add_field("abc", torch.tensor([1, 2]))
    torch.testing.assert_close(batch[0]["abc"], torch.tensor(1))
    torch.testing.assert_close(batch[1]["abc"], torch.tensor(2))
    with pytest.raises(
        AssertionError,
        match="'values' must have the same length as the batch. Got 3 values, whereas the batch has only 2 elements",
    ):
        batch.add_field("bcd", [1, 2, 3])


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
