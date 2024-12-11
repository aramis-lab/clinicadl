import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.extraction import Slice


def test_args():
    with pytest.raises(ValidationError):
        Slice(slices=[0], slice_direction=3)
    with pytest.raises(ValidationError):
        Slice(slices=[0], discarded_slices=[1])
    with pytest.raises(ValidationError):
        Slice(slices=[0], borders=1)


def test_extract_method():
    slice = Slice(slices=[0, 1, 2])
    assert slice.extract_method == "slice"


def test_num_samples_per_image():
    img = torch.randn(1, 5, 7, 3)

    slice = Slice()
    assert slice.num_samples_per_image(img) == 5

    slice = Slice(slices=[1, 2])
    assert slice.num_samples_per_image(img) == 2

    slice = Slice(borders=2, slice_direction=1)
    assert slice.num_samples_per_image(img) == 3

    slice = Slice(discarded_slices=[1, 2], slice_direction=2)
    assert slice.num_samples_per_image(img) == 1

    slice = Slice(discarded_slices=[1], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(img) == 1

    slice = Slice(discarded_slices=[2], borders=2, slice_direction=0)
    assert slice.num_samples_per_image(img) == 0

    slice = Slice(discarded_slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(img)

    slice = Slice(slices=[3], slice_direction=2)
    with pytest.raises(IndexError):
        slice.num_samples_per_image(img)


def test_sample_path():
    slice = Slice(slices=[1, 2, 3])
    assert slice.sample_path(
        Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 1
    ) == Path("sub-001/ses-M000/sub-001_ses-M000_axis-sag_slice-1_T1w.pt")

    slice = Slice(slices=[1, 2, 3], slice_direction=1)
    assert slice.sample_path(
        Path("sub-001/ses-M001/sub-001_ses-M001_FLAIR.nii"), 2
    ) == Path("sub-001/ses-M001/sub-001_ses-M001_axis-cor_slice-2_FLAIR.pt")


def test_extract_sample():
    image_tensor = torch.randn(1, 5, 3, 7)

    slice = Slice(slices=[2, 3])
    assert (
        slice.extract_sample(image_tensor, sample_index=1) == image_tensor[:, 3:4]
    ).all()

    slice = Slice(discarded_slices=[0], slice_direction=1)
    assert (
        slice.extract_sample(image_tensor, sample_index=0) == image_tensor[:, :, 1:2]
    ).all()

    slice = Slice(discarded_slices=[4], borders=1, slice_direction=2)
    assert (
        slice.extract_sample(image_tensor, sample_index=3) == image_tensor[:, :, :, 5:6]
    ).all()

    slice = Slice(discarded_slices=[4], borders=1, slice_direction=2)
    with pytest.raises(IndexError):
        slice.extract_sample(image_tensor, sample_index=4)


def test_extract():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    image_tensor = torch.randn(1, 3, 4, 7)
    image_nifti = nib.Nifti1Image(image_tensor.squeeze(0).numpy(), np.eye(4))
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M000_T1w.nii.gz")

    slice = Slice(discarded_slices=[1, 4], borders=1, slice_direction=2)
    output = slice.extract(tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    assert len(output) == 3
    assert output[0][0] == tmp_dir / "sub-001_ses-M000_axis-axi_slice-0_T1w.pt"
    assert (output[0][1] == image_tensor[:, :, :, 2]).all()
    assert output[1][0] == tmp_dir / "sub-001_ses-M000_axis-axi_slice-1_T1w.pt"
    assert (output[1][1] == image_tensor[:, :, :, 3]).all()
    assert output[2][0] == tmp_dir / "sub-001_ses-M000_axis-axi_slice-2_T1w.pt"
    assert (output[2][1] == image_tensor[:, :, :, 5]).all()

    shutil.rmtree(tmp_dir)


def test_extract_tio_sample():
    slice = Slice(slices=[2, 3])
    image_tensor = torch.randn(1, 5, 7, 3)
    mask_1 = torch.ones(1, 5, 7, 3)
    label = torch.ones(1, 5, 7, 3)

    tio_image = tio.Subject(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
    )
    tio_sample = slice.extract_tio_sample(tio_image, sample_index=1)
    assert isinstance(tio_sample.sample, tio.ScalarImage)
    assert (tio_sample.sample.tensor == image_tensor[:, 3:4]).all()
    assert isinstance(tio_sample.label, tio.LabelMap)
    assert (tio_sample.label.tensor == label[:, 3:4]).all()
    assert isinstance(tio_sample.mask_1, tio.LabelMap)
    assert (tio_sample.mask_1.tensor == mask_1[:, 3:4]).all()
    assert tio_sample.description == 3
    with pytest.raises(AttributeError):
        tio_sample.image

    tio_image = tio.Subject(image=tio.ScalarImage(tensor=image_tensor), label=1)
    tio_sample = slice.extract_tio_sample(tio_image, sample_index=1)
    assert tio_sample.label == 1

    with pytest.raises(IndexError):
        slice.extract_tio_sample(tio_image, sample_index=42)
    with pytest.raises(AttributeError):
        slice.extract_tio_sample(
            tio.Subject(label=tio.LabelMap(tensor=label)), sample_index=1
        )


def test_format_output():
    slice = Slice(slice_direction=2)
    image_tensor = torch.randn(1, 3, 4, 1)
    mask_1 = torch.ones(1, 3, 4, 1)
    label = torch.ones(1, 3, 4, 1)

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
        description=1,
    )
    output = slice.format_output(
        tio_sample,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert (output.sample == image_tensor.squeeze(3)).all()
    assert (output.label == label.squeeze(3)).all()
    assert output.session_id == "ses-M001"
    assert output.participant_id == "sub-001"
    assert output.extraction == "slice"
    assert output.image_path == "sub-001_ses-M001_T1w.nii.gz"
    assert output.slice_direction == 2
    assert output.slice_position == 1

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=0.5,
        description=1,
    )
    output = slice.format_output(
        tio_sample,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert output.label == 0.5

    # check that checks on sample are performed
    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
    )
    with pytest.raises(AttributeError):
        slice.format_output(
            tio_sample,
            participant_id="sub-001",
            session_id="ses-M001",
            image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
        )
