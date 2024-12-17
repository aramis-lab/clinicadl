import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.transforms.extraction import Image


def test_extract_method():
    image = Image()
    assert image.extract_method == "image"


def test_num_samples_per_image():
    image = Image()
    assert image.num_samples_per_image(torch.randn(1, 3, 4, 5)) == 1


def test_sample_path():
    image = Image()
    assert image.sample_path(
        Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 0
    ) == Path("sub-001/ses-M000/sub-001_ses-M000_T1w.pt")
    assert image.sample_path(
        Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.nii"), 0
    ) == Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.pt")


def test_extract_sample():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    assert (image.extract_sample(image_tensor, sample_index=0) == image_tensor).all()

    with pytest.raises(IndexError):
        image.extract_sample(image_tensor, sample_index=1)


def test_extract():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_tensor = torch.randn(1, 3, 4, 5)
    image_nifti = nib.Nifti1Image(image_tensor.squeeze(0).numpy(), np.eye(4))
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M001_FLAIR.nii")

    image = Image()
    output = image.extract(tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    assert len(output) == 1
    assert output[0][0] == tmp_dir / "sub-001_ses-M000_T1w.pt"
    assert (output[0][1] == image_tensor).all()

    output = image.extract(tmp_dir / "sub-001_ses-M001_FLAIR.nii")
    assert output[0][0] == tmp_dir / "sub-001_ses-M001_FLAIR.pt"
    assert (output[0][1] == image_tensor).all()

    shutil.rmtree(tmp_dir)


def test_extract_tio_sample():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    mask_2 = torch.zeros(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    tio_image = tio.Subject(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
        mask_2=tio.LabelMap(tensor=mask_2),
    )
    tio_sample = image.extract_tio_sample(tio_image)
    assert isinstance(tio_sample.sample, tio.ScalarImage)
    assert (tio_sample.sample.tensor == image_tensor).all()
    assert isinstance(tio_sample.label, tio.LabelMap)
    assert (tio_sample.label.tensor == label).all()
    assert isinstance(tio_sample.mask_1, tio.LabelMap)
    assert (tio_sample.mask_1.tensor == mask_1).all()
    assert isinstance(tio_sample.mask_2, tio.LabelMap)
    assert (tio_sample.mask_2.tensor == mask_2).all()
    with pytest.raises(AttributeError):
        tio_sample.image

    tio_image = tio.Subject(image=tio.ScalarImage(tensor=image_tensor), label=1)
    tio_sample = image.extract_tio_sample(tio_image)
    assert tio_sample.label == 1

    with pytest.raises(IndexError):
        image.extract_tio_sample(tio_image, sample_index=1)
    with pytest.raises(AttributeError):
        image.extract_tio_sample(
            tio.Subject(label=tio.LabelMap(tensor=label)), sample_index=1
        )


def test_format_output():
    image = Image()
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
        description=None,
    )
    output = image.format_output(
        tio_sample,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert (output.sample == image_tensor).all()
    assert (output.label == label).all()
    assert output.session_id == "ses-M001"
    assert output.participant_id == "sub-001"
    assert output.extraction == "image"
    assert output.image_path == "sub-001_ses-M001_T1w.nii.gz"

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=0.5,
        description=None,
    )
    output = image.format_output(
        tio_sample,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert output.label == 0.5


@pytest.mark.parametrize(
    "tio_sample",
    [
        tio.Subject(
            label=tio.LabelMap(tensor=torch.ones(1, 3, 4, 5)),
            description=None,
        ),
        tio.Subject(
            sample=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
            description=None,
        ),
        tio.Subject(
            label=0.5,
            sample=tio.ScalarImage(tensor=torch.randn(1, 3, 4, 5)),
        ),
    ],
)
def test_format_output_errors(tio_sample):
    image = Image()
    with pytest.raises(AttributeError):
        image.format_output(
            tio_sample,
            participant_id="sub-001",
            session_id="ses-M001",
            image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
        )
