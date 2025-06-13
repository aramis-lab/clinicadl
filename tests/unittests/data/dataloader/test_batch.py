import torch
import torchio as tio

from clinicadl.data.dataloader.batch import SimpleBatch
from clinicadl.data.datatypes import T1Linear
from clinicadl.transforms.extraction.image import ImageSample


def test_SimpleBatch():
    list_samples = [
        ImageSample(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            participant=f"sub-{i}",
            session=f"ses-{i}",
            preprocessing=T1Linear(),
            image_path=f"{i}.pt",
        )
        for i in range(3)
    ]
    batch = SimpleBatch(list_samples)
    images = batch.get_images()
    assert images.size() == (3, 1, 3, 4, 5)
    labels = batch.get_labels()
    assert labels.size() == (3, 1, 3, 4, 5)

    batch.append(
        ImageSample(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
            label=1,
            participant="sub-3",
            session="ses-3",
            preprocessing=T1Linear(),
            image_path="3.pt",
        )
    )
    images = batch.get_images()
    assert isinstance(images, list)
    assert images[0].size() == (1, 3, 4, 5)
    assert images[-1].size() == (1, 3, 4, 6)
    labels = batch.get_labels()
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1] == 1

    batch[-1] = ImageSample(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 6)),
        label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 6)),
        participant="sub-3",
        session="ses-3",
        preprocessing=T1Linear(),
        image_path="3.pt",
    )
    labels = batch.get_labels()
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1].size() == (1, 3, 4, 6)

    batch[-1] = ImageSample(
        image=tio.ScalarImage(tensor=torch.ones(1, 3, 4, 6)),
        label=None,
        participant="sub-3",
        session="ses-3",
        preprocessing=T1Linear(),
        image_path="3.pt",
    )
    labels = batch.get_labels()
    assert isinstance(labels, list)
    assert labels[0].size() == (1, 3, 4, 5)
    assert labels[-1] is None
