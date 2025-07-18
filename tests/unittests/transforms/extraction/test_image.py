import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.extraction import Image


def test_extract_method():
    image = Image()
    assert image.extract_method == "image"


def test_num_samples_per_image():
    image = Image()
    assert image.num_samples_per_image(torch.randn(1, 3, 4, 5)) == 1


def test_extract_sample():
    image = Image()
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.randint(0, 2, (1, 3, 4, 5))
    label = torch.randint(0, 2, (2, 3, 4, 5))

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        participant="sub-000",
        session="ses-M000",
        image_path="abc.nii.gz",
        label=tio.LabelMap(tensor=label, affine=affine),
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    extracted_data_point = image.extract_sample(data_point)
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == image_tensor).all()
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == label).all()
    assert isinstance(extracted_data_point["mask_1"], tio.LabelMap)
    assert (extracted_data_point["mask_1"].tensor == mask_1).all()

    assert np.isclose(extracted_data_point.image.affine, affine).all()
    assert np.isclose(extracted_data_point.label.affine, affine).all()

    assert extracted_data_point.participant == "sub-000"
    assert extracted_data_point.session == "ses-M000"
    assert extracted_data_point.image_path == "abc.nii.gz"
    assert extracted_data_point._sample_index == 1

    # other tests
    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=1,
        participant="sub-000",
        session="ses-M000",
    )
    extracted_data_point = image.extract_sample(data_point)
    assert extracted_data_point.label == 1

    with pytest.raises(IndexError):
        image.extract_sample(data_point, sample_index=1)

    # test transforms history
    transform = tio.Clamp(out_min=0, out_max=10)
    sample = image.extract_sample(transform(data_point))
    assert len(sample.get_applied_transforms()) == 1
    assert isinstance(sample.get_applied_transforms()[0], tio.Clamp)
