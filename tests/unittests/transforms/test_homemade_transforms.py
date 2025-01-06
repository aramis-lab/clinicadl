import torch
import torchio as tio

from clinicadl.transforms.homemade_transforms import NanRemoval


def test_nan_removal():
    img = torch.zeros(1, 4, 5, 3)
    label = torch.ones(1, 4, 5, 3)

    label[0, 0, 0, 0] = torch.nan

    img[0, 0, 0, 0] = torch.nan
    img[0, 1, 1, 0] = torch.inf
    img[0, 1, 0, 0] = -torch.inf

    sub = tio.Subject(
        image=tio.ScalarImage(tensor=img), label=tio.LabelMap(tensor=label)
    )

    transform = NanRemoval()
    transformed = transform(sub)
    assert transformed.image.tensor[0, 0, 0, 0] == 0
    assert transformed.image.tensor[0, 1, 1, 0] == torch.finfo().max
    assert transformed.image.tensor[0, 1, 0, 0] == torch.finfo().min
    assert transformed.label.tensor[0, 0, 0, 0].isnan()

    transform = NanRemoval(nan=1, posinf=1, neginf=-1)
    transformed = transform(sub)
    assert transformed.image.tensor[0, 0, 0, 0] == 1
    assert transformed.image.tensor[0, 1, 1, 0] == 1
    assert transformed.image.tensor[0, 1, 0, 0] == -1
