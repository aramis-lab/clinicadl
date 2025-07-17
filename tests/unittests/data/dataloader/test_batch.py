import torch
import torchio as tio

from clinicadl.data.dataloader.batch import SimpleBatch
from clinicadl.data.structures import DataPoint


def test_SimpleBatch():
    # tensor labels
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = SimpleBatch(list_samples)
    images = batch.get_images()
    assert images.size() == (2, 1, 3, 4, 5)
    labels = batch.get_labels()
    assert labels.size() == (2, 1, 3, 4, 5)

    # different shapes
    batch.append(
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 6)),
            participant="sub-2",
            session="ses-2",
        )
    )
    labels = batch.get_labels()
    images = batch.get_images()
    assert isinstance(images, list)
    assert images[0].size() == (1, 3, 4, 5)
    assert images[-1].size() == (1, 3, 4, 6)
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1].size() == (1, 3, 4, 6)

    # heterogeneous labels
    batch[-1] = DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
        label=1,
        participant="sub-2",
        session="ses-2",
    )
    labels = batch.get_labels()
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1] == 1

    # a None label
    batch[-1] = DataPoint(
        image=tio.ScalarImage(tensor=torch.ones(1, 3, 4, 6)),
        label=None,
        participant="sub-2",
        session="ses-2",
    )
    labels = batch.get_labels()
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1] is None

    # a dict label
    batch[-1] = DataPoint(
        image=tio.ScalarImage(tensor=torch.ones(1, 3, 4, 6)),
        label={"A": 0, "B": 1},
        participant="sub-2",
        session="ses-2",
    )
    labels = batch.get_labels()
    assert isinstance(labels, list)
    torch.testing.assert_close(
        labels[-1], torch.tensor([0.0, 1.0], dtype=torch.float32)
    )

    # all int labels
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=i,
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = SimpleBatch(list_samples)
    labels = batch.get_labels()
    torch.testing.assert_close(labels, torch.tensor([0, 1], dtype=torch.int64))

    # all float in labels
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=float(i),
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = SimpleBatch(list_samples)
    labels = batch.get_labels()
    torch.testing.assert_close(labels, torch.tensor([0.0, 1.0], dtype=torch.float32))

    # all dict in labels
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label={"A": 0, "B": 1},
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(2)
    ]
    batch = SimpleBatch(list_samples)
    labels = batch.get_labels()
    torch.testing.assert_close(
        labels, torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    )

    # without tensors
    list_samples = [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label={"A": 0, "B": 1} if i == 0 else 1 if i == 1 else None,
            participant=f"sub-{i}",
            session=f"ses-{i}",
        )
        for i in range(3)
    ]
    batch = SimpleBatch(list_samples)
    labels = batch.get_labels()
    assert isinstance(labels, list)
    torch.testing.assert_close(labels[0], torch.tensor([0.0, 1.0], dtype=torch.float32))
    assert labels[1] == 1
    assert labels[2] is None
