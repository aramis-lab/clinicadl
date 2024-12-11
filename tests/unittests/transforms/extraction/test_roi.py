import shutil
from pathlib import Path
from typing import Tuple, Union

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.transforms.extraction import ROI


@pytest.mark.skip
def _generate_random_mask(
    size: Tuple[int, ...], path: Union[str, Path], max_value: int = 1
):
    path = Path(path)
    mask = np.random.randint(0, max_value + 1, size).astype(float)
    image_nifti = nib.Nifti1Image(mask, np.eye(4))
    nib.save(image_nifti, path)


@pytest.mark.skip
def _mask_from_numpy(mask_np: np.ndarray, path: Union[str, Path]):
    image_nifti = nib.Nifti1Image(mask_np, np.eye(4))
    nib.save(image_nifti, path)


def test_args():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        ROI(masks=[])

    with pytest.raises(FileNotFoundError):
        ROI(masks=[tmp_dir / "abc.nii.gz"])

    mask = np.random.randint(0, 2, (4, 7, 6))
    torch.save(torch.from_numpy(mask), tmp_dir / "mask.pt")
    with pytest.raises(Exception):
        ROI(masks=[tmp_dir / "mask.pt"])

    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask.nii.gz", max_value=2)
    with pytest.raises(ValueError):
        ROI(masks=[tmp_dir / "mask.nii.gz"])

    _generate_random_mask(size=(7, 6), path=tmp_dir / "mask.nii.gz")
    with pytest.raises(ValueError):
        ROI(masks=[tmp_dir / "mask.nii.gz"])

    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_1.nii.gz")
    _generate_random_mask(size=(3, 7, 6), path=tmp_dir / "mask_2.nii.gz")
    with pytest.raises(ValueError):
        ROI(masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"])

    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_2.nii.gz")
    ROI(masks=[tmp_dir / "mask_1.nii.gz"])
    ROI(masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"])

    shutil.rmtree(tmp_dir)


def test_num_samples_per_image():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_1.nii.gz")
    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_2.nii.gz")
    img = torch.randn(1, 5, 7, 3)

    assert ROI(masks=[tmp_dir / "mask_1.nii.gz"]).num_samples_per_image(img) == 1
    assert (
        ROI(
            masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"]
        ).num_samples_per_image(img)
        == 2
    )

    assert ROI(masks=[tmp_dir / "mask_1.nii.gz"]).extract_method == "roi"

    shutil.rmtree(tmp_dir)


def test_sample_path():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_1.nii.gz")
    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_2.nii.gz")

    assert ROI(masks=[tmp_dir / "mask_1.nii.gz"]).sample_path(
        Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 0
    ) == Path("sub-001/ses-M000/sub-001_ses-M000_roi-mask_1_T1w.pt")

    assert ROI(
        masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"]
    ).sample_path(Path("sub-001/ses-M000/sub-001_ses-M000_T1w.nii.gz"), 1) == Path(
        "sub-001/ses-M000/sub-001_ses-M000_roi-mask_2_T1w.pt"
    )

    shutil.rmtree(tmp_dir)


def test_extract_sample():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_tensor = torch.randn(1, 9, 10, 9)

    mask_1 = np.zeros((9, 10, 9))
    mask_1[3:6, 3:5, 4:6] = 1
    mask_1_tensor = torch.from_numpy(mask_1).unsqueeze(0).int()
    _mask_from_numpy(mask_1, path=tmp_dir / "mask_1.nii.gz")

    mask_2 = np.zeros((9, 10, 9))
    mask_2[6:, 5:7, 1:3] = 1
    mask_2_tensor = torch.from_numpy(mask_2).unsqueeze(0).int()
    _mask_from_numpy(mask_2, path=tmp_dir / "mask_2.nii.gz")

    roi = ROI(masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"])
    assert torch.isclose(
        roi.extract_sample(image_tensor, sample_index=0).sum(),
        (image_tensor * mask_1_tensor).sum(),
    )
    assert roi.extract_sample(image_tensor, sample_index=0).shape == (1, 9, 10, 9)
    assert torch.isclose(
        roi.extract_sample(image_tensor, sample_index=1).sum(),
        (image_tensor * mask_2_tensor).sum(),
    )
    assert roi.extract_sample(image_tensor, sample_index=1).shape == (1, 9, 10, 9)

    roi = ROI(
        masks=[tmp_dir / "mask_1.nii.gz", str(tmp_dir / "mask_2.nii.gz")], crop=True
    )
    assert torch.isclose(
        roi.extract_sample(image_tensor, sample_index=0).sum(),
        (image_tensor * mask_1_tensor).sum(),
    )
    assert roi.extract_sample(image_tensor, sample_index=0).shape == (1, 6, 4, 5)
    assert torch.isclose(
        roi.extract_sample(image_tensor, sample_index=1).sum(),
        (image_tensor * mask_2_tensor).sum(),
    )
    assert roi.extract_sample(image_tensor, sample_index=1).shape == (1, 6, 4, 5)
    assert roi.output_size == (6, 4, 5)

    with pytest.raises(IndexError):
        roi.extract_sample(image_tensor, sample_index=2)
    with pytest.raises(ValueError):
        roi.extract_sample(torch.randn(1, 8, 10, 9).float(), sample_index=0)

    shutil.rmtree(tmp_dir)


def test_extract():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mask_1 = np.zeros((9, 10, 9))
    mask_1[3:6, 3:5, 4:6] = 1
    mask_1_tensor = torch.from_numpy(mask_1).unsqueeze(0).int()
    _mask_from_numpy(mask_1, path=tmp_dir / "mask_1.nii.gz")

    mask_2 = np.zeros((9, 10, 9))
    mask_2[6:, 5:7, 1:3] = 1
    mask_2_tensor = torch.from_numpy(mask_2).unsqueeze(0).int()
    _mask_from_numpy(mask_2, path=tmp_dir / "mask_2.nii.gz")

    image_tensor = torch.randn(1, 9, 10, 9).float()
    image_nifti = nib.Nifti1Image(image_tensor.squeeze(0).numpy(), np.eye(4))
    nib.save(image_nifti, tmp_dir / "sub-001_ses-M000_T1w.nii.gz")

    output = ROI(
        masks=[tmp_dir / "mask_1.nii.gz", tmp_dir / "mask_2.nii.gz"], crop=True
    ).extract(tmp_dir / "sub-001_ses-M000_T1w.nii.gz")
    assert len(output) == 2
    assert output[0][0] == tmp_dir / "sub-001_ses-M000_roi-mask_1_T1w.pt"
    assert torch.isclose(output[0][1].sum(), (image_tensor * mask_1_tensor).sum())
    assert output[0][1].shape == (1, 6, 4, 5)
    assert output[1][0] == tmp_dir / "sub-001_ses-M000_roi-mask_2_T1w.pt"
    assert torch.isclose(output[1][1].sum(), (image_tensor * mask_2_tensor).sum())
    assert output[1][1].shape == (1, 6, 4, 5)

    shutil.rmtree(tmp_dir)


def test_extract_tio_sample():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_tensor = torch.randn(1, 9, 10, 9)
    label = torch.ones(1, 9, 10, 9)
    mask_1 = torch.zeros(1, 9, 10, 9)

    mask_roi = np.zeros((9, 10, 9))
    mask_roi[3:6, 3:5, 4:6] = 1
    mask_roi_tensor = torch.from_numpy(mask_roi).unsqueeze(0).int()
    _mask_from_numpy(mask_roi, path=tmp_dir / "mask_1.nii.gz")

    roi = ROI(masks=[tmp_dir / "mask_1.nii.gz"])

    tio_image = tio.Subject(
        image=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
    )
    tio_sample = roi.extract_tio_sample(tio_image, sample_index=0)
    assert isinstance(tio_sample.sample, tio.ScalarImage)
    assert torch.isclose(
        tio_sample.sample.tensor.sum(), (image_tensor * mask_roi_tensor).sum()
    )
    assert isinstance(tio_sample.label, tio.LabelMap)
    assert torch.isclose(tio_sample.label.tensor.sum(), (label * mask_roi_tensor).sum())
    assert isinstance(tio_sample.mask_1, tio.LabelMap)
    assert torch.isclose(
        tio_sample.mask_1.tensor.sum(), (mask_1 * mask_roi_tensor).sum()
    )
    assert tio_sample.description == str(tmp_dir / "mask_1.nii.gz")
    with pytest.raises(AttributeError):
        tio_sample.image

    tio_image = tio.Subject(image=tio.ScalarImage(tensor=image_tensor), label=1)
    tio_sample = roi.extract_tio_sample(tio_image, sample_index=0)
    assert tio_sample.label == 1

    with pytest.raises(IndexError):
        roi.extract_tio_sample(tio_image, sample_index=1)
    with pytest.raises(AttributeError):
        roi.extract_tio_sample(
            tio.Subject(label=tio.LabelMap(tensor=label)), sample_index=0
        )


def test_format_output():
    tmp_dir = Path(__file__).parents[2] / "ressources" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    _generate_random_mask(size=(4, 7, 6), path=tmp_dir / "mask_1.nii.gz")
    roi = ROI(masks=[tmp_dir / "mask_1.nii.gz"])

    image_tensor = torch.randn(1, 4, 7, 6)
    mask_1 = torch.ones(1, 4, 7, 6)
    label = torch.ones(1, 4, 7, 6)

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=tio.LabelMap(tensor=label),
        mask_1=tio.LabelMap(tensor=mask_1),
        description=str(tmp_dir / "mask_1.nii.gz"),
    )
    output = roi.format_output(
        tio_sample,
        participant_id="sub-001",
        session_id="ses-M001",
        image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
    )
    assert (output.sample == image_tensor).all()
    assert (output.label == label).all()
    assert output.session_id == "ses-M001"
    assert output.participant_id == "sub-001"
    assert output.extraction == "roi"
    assert output.image_path == "sub-001_ses-M001_T1w.nii.gz"
    assert output.roi == str(tmp_dir / "mask_1.nii.gz")
    assert not output.cropped

    tio_sample = tio.Subject(
        sample=tio.ScalarImage(tensor=image_tensor),
        label=0.5,
        description=str(tmp_dir / "mask_1.nii.gz"),
    )
    output = roi.format_output(
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
        roi.format_output(
            tio_sample,
            participant_id="sub-001",
            session_id="ses-M001",
            image_path=Path("sub-001_ses-M001_T1w.nii.gz"),
        )
