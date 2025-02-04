from pathlib import Path

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
    mask_1 = torch.ones(1, 3, 4, 5)
    mask_2 = torch.zeros(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        participant="sub-000",
        session="ses-M000",
        label=tio.LabelMap(tensor=label, affine=affine),
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
        mask_2=tio.LabelMap(tensor=mask_2, affine=affine),
    )
    extracted_data_point, description = image.extract_sample(data_point)
    assert description is None
    assert isinstance(extracted_data_point.image, tio.ScalarImage)
    assert (extracted_data_point.image.tensor == image_tensor).all()
    assert isinstance(extracted_data_point.label, tio.LabelMap)
    assert (extracted_data_point.label.tensor == label).all()
    assert isinstance(extracted_data_point.mask_1, tio.LabelMap)
    assert (extracted_data_point.mask_1.tensor == mask_1).all()
    assert isinstance(extracted_data_point.mask_2, tio.LabelMap)
    assert (extracted_data_point.mask_2.tensor == mask_2).all()

    assert np.isclose(extracted_data_point.image.affine, affine).all()
    assert np.isclose(extracted_data_point.label.affine, affine).all()

    data_point = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor),
        label=1,
        participant="sub-000",
        session="ses-M000",
    )
    extracted_data_point, _ = image.extract_sample(data_point)
    assert extracted_data_point.label == 1

    with pytest.raises(IndexError):
        image.extract_sample(data_point, sample_index=1)


def test_format_output():
    image = Image()
    affine = np.diag([3, 2, 1, 1])
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        participant="sub-000",
        session="ses-M000",
        label=tio.LabelMap(tensor=label, affine=affine),
        mask_1=tio.LabelMap(tensor=mask_1, affine=affine),
    )
    output = image.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.pt"),
    )
    assert (output.sample == image_tensor).all()
    assert (output.label == label).all()
    assert np.isclose(output.affine, affine).all()
    assert output.session == "ses-M000"
    assert output.participant == "sub-000"
    assert output.extraction == "image"
    assert output.image_path == "sub-000_ses-M000_T1w.pt"

    sample_data = DataPoint(
        image=tio.ScalarImage(tensor=image_tensor, affine=affine),
        participant="sub-000",
        session="ses-M000",
        label=0.5,
    )
    output = image.format_output(
        sample_data,
        image_path=Path("sub-000_ses-M000_T1w.nii.gz"),
    )
    assert output.label == 0.5
