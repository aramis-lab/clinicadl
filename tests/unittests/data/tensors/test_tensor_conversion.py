import json
import shutil
import warnings
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.data.structures import DataPoint, Mask
from clinicadl.data.tensors import TensorConversion
from clinicadl.transforms.config import (
    ClampConfig,
    CropConfig,
    PadConfig,
    RescaleIntensityConfig,
    ToCanonicalConfig,
)
from clinicadl.transforms.extraction import Slice
from clinicadl.transforms.handlers import Transforms
from clinicadl.utils.exceptions import (
    CannotReadJsonFieldError,
    MissingFieldsJsonError,
    TensorConversionError,
)

DATASET_DIR = Path(__file__).parents[2] / "resources" / "caps_example"


class CustomTransform(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint.add_image(datapoint.image, "other_image")
        datapoint.add_mask(
            tio.LabelMap(
                tensor=torch.ones_like(datapoint.image.tensor),
                affine=datapoint.image.affine,
            ),
            "other_mask",
        )
        datapoint["coefficient"] = 0.5

        return datapoint


class CustomTransformBis(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint.add_image(datapoint.image, "other_image")
        datapoint["other_mask"] = 1.0
        datapoint["coefficient"] = 0.5

        return datapoint


class TensorDataset:
    def __init__(self):
        self.config = Mock()
        self.config.directory = None
        self.config.datatype = PET_DATATYPE
        self.config.transforms = Transforms()

        self.individual_masks = []
        self.common_masks = []

        self.sub_ses = []

    @property
    def _tensor_conversion_json_dir(self) -> Path:
        return self.config.directory / "tensor_conversion"

    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        return set(self.sub_ses)

    def _get_image_path(self, participant: str, session: str) -> Path:
        pre_path = self.config.directory / "subjects" / participant / session
        if isinstance(self.config.datatype, PETLinear):
            file_path = (
                pre_path
                / "pet_linear"
                / f"{participant}_{session}_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz"
            )
        else:
            file_path = (
                pre_path
                / "t1_linear"
                / f"{participant}_{session}_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
            )

        return file_path

    def _get_common_mask_path(self, mask_name: str) -> Path:
        return self.config.directory / "masks" / mask_name


PET_DATATYPE = PETLinear(
    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
)
T1_DATATYPE = T1Linear(use_uncropped_image=True)


def test_convert_and_read(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)

    dataset = TensorDataset()
    dataset.config.directory = tmp_path
    dataset.sub_ses = [
        ("sub-100", "ses-M000"),
    ]

    converter = TensorConversion(dataset)
    converter.to_tensors(conversion_name="pet_tmp")
    converter.read_tensor_conversion(conversion_name="pet_tmp")

    shutil.rmtree(tmp_path)
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    shutil.rmtree(tmp_path / "tensor_conversion")

    converter.to_tensors()
    converter.read_tensor_conversion()


def test_read_tensor_conversion():
    dataset = TensorDataset()
    dataset.config.directory = DATASET_DIR
    sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-000", "ses-M003"),
        ("sub-010", "ses-M003"),
        ("sub-010", "ses-M012"),
    ]
    dataset.sub_ses = sub_ses

    converter = TensorConversion(dataset)
    converter.read_tensor_conversion()
    assert str(converter.json) == str(
        DATASET_DIR / "tensor_conversion" / "default_pet-linear_18FAV45_pons2.json"
    )
    info = converter.get_info()
    assert info.datatype == PET_DATATYPE
    assert info.individual_masks == []
    assert info.common_masks == []
    assert info.transforms == []
    assert info.spacing is None
    assert info.shape == (1, 1, 1, 1)
    assert sorted(info.participants_sessions) == sorted(sub_ses)

    # test tensor path and json name
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion("default_pet-linear_18FAV45_pons2")
    assert str(converter.json) == str(
        DATASET_DIR / "tensor_conversion" / "default_pet-linear_18FAV45_pons2.json"
    )
    assert converter.conversion_name == "default_pet-linear_18FAV45_pons2"
    assert converter.tensors_path == Path("tensors/default_pet-linear_18FAV45_pons2")

    converter.read_tensor_conversion("pet_small", check_pt_files=False)
    assert str(converter.json) == str(
        DATASET_DIR / "tensor_conversion" / "pet_small.json"
    )
    assert converter.conversion_name == "pet_small"

    # check pt files
    dataset.config.datatype = T1_DATATYPE
    dataset.sub_ses = [("sub-010", "ses-M012")]
    converter = TensorConversion(dataset)
    with pytest.raises(
        FileNotFoundError,
        match="Tensor conversion was performed, as suggested by the presence of*",
    ):
        converter.read_tensor_conversion(conversion_name="t1_missing_session")

    dataset.sub_ses = [("sub-000", "ses-M000")]
    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "leftHemisphere.nii")]
    converter = TensorConversion(dataset)
    with pytest.raises(
        FileNotFoundError,
        match="Tensor conversion was performed, as suggested by the presence of*",
    ):
        converter.read_tensor_conversion(conversion_name="t1_missing_mask")

    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "leftHippocampus.nii.gz")]
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion(conversion_name="t1_masks")

    # check json
    with pytest.raises(FileNotFoundError):
        converter.read_tensor_conversion(conversion_name="abc")
    with pytest.raises(
        CannotReadJsonFieldError,
        match=r"TensorConversionInfo cannot read the field\(s\) \['transforms'\] in .*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_ref_corrupted")
    with pytest.raises(
        CannotReadJsonFieldError,
        match=r"TensorConversionInfo cannot read the field\(s\) \['transforms'\] in .*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_ref_corrupted_bis")
    with pytest.raises(
        MissingFieldsJsonError,
        match=r"Fields \['datatype'\] are missing in .*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_ref_missing_field")

    # check preprocessing
    dataset.common_masks = []
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="The datatype of the old conversion does not match the current datatype.*",
    ):
        converter.read_tensor_conversion("default_pet-linear_18FAV45_pons2")

    # check masks
    dataset.config.datatype = PET_DATATYPE
    dataset.individual_masks = [Mask("brain"), Mask("seg")]
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])

    dataset.individual_masks = []
    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "leftHippocampus.nii.gz")]
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    dataset.individual_masks = [Mask("brain"), Mask("seg")]
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    dataset.individual_masks = [Mask("brain"), Mask("seg"), Mask("hippocampus")]
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="Some image-specific masks have not been converted*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_masks")

    dataset.individual_masks = []
    dataset.common_masks = (
        Mask(DATASET_DIR / "masks" / "leftHippocampus.nii.gz"),
        Mask(DATASET_DIR / "masks" / "rightHemisphere.nii"),
    )
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError, match="Some masks have not been converted*"
    ):
        converter.read_tensor_conversion(conversion_name="pet_masks")

    # check also
    dataset.common_masks = []
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="You asked 'abc' in 'load_also', but no such information was stored during conversion*",
    ):
        converter.read_tensor_conversion(
            conversion_name="pet_custom_transform",
            load_also=["other_image", "abc"],
            check_transforms=False,
        )
    converter.read_tensor_conversion(
        conversion_name="pet_custom_transform",
        load_also=["other_image", "coefficient"],
        check_transforms=False,
        check_pt_files=False,
    )

    # check transforms
    dataset.config.transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[
            RescaleIntensityConfig(),
            ClampConfig(out_min=-10, out_max=10),
        ],
        sample_transforms=[CropConfig(cropping=1)],
        augmentations=[PadConfig(padding=1)],
    )
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion("pet_transform", check_pt_files=False)
    assert len(converter.get_info().transforms) == 2
    with pytest.raises(
        TensorConversionError,
        match="Custom transforms have been used during the old conversion*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_custom_transform")
    converter.read_tensor_conversion(
        "pet_custom_transform", check_transforms=False, check_pt_files=False
    )
    assert len(converter.get_info().transforms) == 2

    dataset.config.transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[
            RescaleIntensityConfig(),
            ToCanonicalConfig(),
        ],
        sample_transforms=[CropConfig(cropping=1)],
        augmentations=[PadConfig(padding=1)],
    )
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="The image transforms applied during the old conversion*",
    ):
        converter.read_tensor_conversion(
            "pet_transform",
        )

    dataset.config.transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[
            RescaleIntensityConfig(),
            tio.Clamp(out_min=-10, out_max=10),
        ],
        sample_transforms=[CropConfig(cropping=1)],
        augmentations=[PadConfig(padding=1)],
    )
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="Custom transforms have been passed to the dataset*",
    ):
        converter.read_tensor_conversion(conversion_name="pet_transform")

    converter.read_tensor_conversion(conversion_name="default_pet-linear_18FAV45_pons2")
    assert converter.get_info().transforms == []

    # check subject session
    dataset.config.transforms = Transforms()
    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-000", "ses-M003"),
        ("sub-010", "ses-M003"),
    ]
    converter = TensorConversion(dataset)
    converter.read_tensor_conversion()

    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-000", "ses-M003"),
        ("sub-999", "ses-M999"),
    ]
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r"Some \(participant, session\) have not been converted*",
    ):
        converter.read_tensor_conversion("pet_small")


def test_to_tensors(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)

    dataset = TensorDataset()
    dataset.config.directory = tmp_path

    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    dataset.config.datatype = T1_DATATYPE

    # control
    dataset.individual_masks = [Mask("brain"), Mask("seg")]
    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "leftHippocampus.nii.gz")]
    dataset.config.transforms = Transforms(
        image_transforms=[
            CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
            tio.Clamp(out_max=10),
        ]
    )
    converter = TensorConversion(dataset)
    converter.to_tensors(
        conversion_name="new_conversion", n_proc=2, save_transforms=True
    )
    with open(tmp_path / "tensor_conversion" / "new_conversion.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["datatype"] == {
        "name": "T1Linear",
        "pattern": "t1_linear/sub-.*_ses-.*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.*",
        "key": "t1-linear",
        "description": "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline",
        "use_uncropped_image": True,
    }
    assert set(conversion_info["individual_masks"]) == {"brain", "seg"}
    assert conversion_info["common_masks"] == ["leftHippocampus.nii.gz"]
    assert conversion_info["transforms"] == [
        {
            "name": "Crop",
            "exclude": None,
            "include": None,
            "cropping": [0, 1, 0, 1, 0, 1],
        },
        "Clamp(out_min=None, out_max=10)",
    ]
    assert np.isclose(conversion_info["spacing"], [1.3, 1.2, 1.1]).all()
    assert conversion_info["shape"] == [1, 2, 2, 2]
    assert not conversion_info["interrupted"]
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )
    print(conversion_info)
    #       check files
    assert (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "new_conversion"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    assert (
        tmp_path / "masks" / "tensors" / "new_conversion" / "leftHippocampus.pt"
    ).is_file()

    tensors: dict = torch.load(
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "new_conversion"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    true_tensors: dict = torch.load(
        DATASET_DIR
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_transform"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(list(tensors.keys())) == sorted(["image", "seg", "brain", "affine"])
    for name in tensors.keys():
        assert (tensors[name] == true_tensors[name]).all()

    # test without saving transforms
    dataset.individual_masks = []
    dataset.common_masks = []
    dataset.config.transforms = Transforms(
        image_transforms=[
            CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
        ]
    )

    converter = TensorConversion(dataset)
    converter.to_tensors()
    with open(tmp_path / "tensor_conversion" / "default_t1-linear.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["transforms"] == []
    tensors = torch.load(
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "default_t1-linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert tensors["image"].shape == (1, 3, 3, 3)  # not cropped

    # spacing checked
    dataset.config.transforms = Transforms()
    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M012"),
    ]

    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match="An error occurred during conversion.*",
    ):
        converter.to_tensors(conversion_name="check_spacing")
    with open(tmp_path / "tensor_conversion" / "check_spacing.json", "r") as f:
        conversion_info = json.load(f)
    assert len(conversion_info["participants_sessions"]) == 1
    assert conversion_info["spacing"] is not None
    assert (
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "check_spacing"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file() != (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "check_spacing"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    converter.to_tensors(conversion_name="no_check_spacing", ignore_spacing=True)
    with open(tmp_path / "tensor_conversion" / "no_check_spacing.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["spacing"] is None
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M012"],
        ]
    )
    assert (
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "no_check_spacing"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "no_check_spacing"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    # shape warning
    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M012"),
    ]

    converter = TensorConversion(dataset)
    with pytest.warns(match="Different image shapes found in the dataset:*"):
        converter.to_tensors(conversion_name="check_shape", ignore_spacing=True)
    with open(tmp_path / "tensor_conversion" / "check_shape.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["shape"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.to_tensors(
            conversion_name="not_check_shape", ignore_spacing=True, shape_warning=False
        )

    # check consistency in subjects
    dataset.sub_ses = [
        ("sub-000", "ses-M003"),
    ]
    dataset.individual_masks = [Mask("seg")]

    converter = TensorConversion(dataset)
    with pytest.raises(TensorConversionError):
        converter.to_tensors(conversion_name="subject_consistency")

    dataset.individual_masks = [Mask("brain")]
    converter = TensorConversion(dataset)
    with pytest.raises(TensorConversionError):
        converter.to_tensors(conversion_name="subject_consistency")

    dataset.individual_masks = []
    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "rightHemisphere.nii")]
    converter = TensorConversion(dataset)
    with pytest.raises(TensorConversionError):
        converter.to_tensors(conversion_name="subject_consistency")

    dataset.common_masks = [Mask(DATASET_DIR / "masks" / "rightHippocampus.nii.gz")]
    converter = TensorConversion(dataset)
    with pytest.raises(TensorConversionError):
        converter.to_tensors(conversion_name="subject_consistency")
    converter.to_tensors(conversion_name="subject_consistency", ignore_spacing=True)

    dataset.common_masks = []
    converter = TensorConversion(dataset)
    converter.to_tensors(conversion_name="subject_consistency_control")

    # custom transform
    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    dataset.config.transforms = Transforms(
        image_transforms=[CustomTransform()],
    )
    dataset.individual_masks = [Mask("brain"), Mask("seg")]

    converter = TensorConversion(dataset)
    converter.to_tensors(conversion_name="custom_transform", save_transforms=True)
    with open(tmp_path / "tensor_conversion" / "custom_transform.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["individual_masks"]) == sorted(["brain", "seg"])
    assert conversion_info["also"] == {
        "other_image": "image",
        "other_mask": "mask",
        "coefficient": "other",
    }
    tensors: dict = torch.load(
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "custom_transform"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(tensors.keys()) == sorted(
        [
            "image",
            "seg",
            "brain",
            "coefficient",
            "other_image",
            "other_mask",
            "affine",
        ]
    )
    assert tensors["coefficient"] == 0.5
    assert tensors["other_image"].dtype == torch.float32
    assert tensors["other_mask"].dtype == torch.int32

    converter = TensorConversion(dataset)
    converter.to_tensors(
        conversion_name="custom_transform_not_saved", save_transforms=False
    )
    with open(
        tmp_path / "tensor_conversion" / "custom_transform_not_saved.json", "r"
    ) as f:
        conversion_info = json.load(f)
    assert conversion_info["also"] == {}
    assert sorted(
        torch.load(
            tmp_path
            / "subjects"
            / "sub-000"
            / "ses-M000"
            / "t1_linear"
            / "tensors"
            / "custom_transform_not_saved"
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
            weights_only=True,
        ).keys()
    ) == sorted(
        [
            "image",
            "seg",
            "brain",
            "affine",
        ]
    )

    # check conversion name
    converter = TensorConversion(dataset)
    with pytest.raises(ValueError, match="'conversion_name' can't start with default"):
        converter.to_tensors(conversion_name="default_conversion")
    with pytest.raises(
        ValueError,
        match="If 'save_transforms' is True, 'conversion_name' cannot be None.",
    ):
        converter.to_tensors(save_transforms=True)


def test_overwrite(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)

    dataset = TensorDataset()
    dataset.config.directory = tmp_path

    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
    ]
    dataset.config.datatype = T1_DATATYPE
    dataset.individual_masks = [Mask("brain")]

    converter = TensorConversion(dataset)

    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge*",
    ):
        converter.to_tensors()

    converter.to_tensors(overwrite=True)

    with open(tmp_path / "tensor_conversion" / "default_t1-linear.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["individual_masks"] == ["brain"]
    tensors = torch.load(
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "default_t1-linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert "brain" in tensors

    converter.to_tensors(conversion_name="t1_masks", overwrite=True)

    with open(tmp_path / "tensor_conversion" / "t1_masks.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["individual_masks"] == ["brain"]
    tensors = torch.load(
        tmp_path
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert "seg" not in tensors


def test_merge_conversions(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    dataset = TensorDataset()
    dataset.config.directory = tmp_path

    dataset.sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    dataset.config.datatype = T1_DATATYPE

    # control
    converter = TensorConversion(dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.to_tensors(conversion_name="t1_ref_interrupted")
    with open(tmp_path / "tensor_conversion" / "t1_ref_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )
    assert np.isclose(conversion_info["spacing"], (1.3, 1.2, 1.1)).all()
    assert conversion_info["shape"] == [1, 3, 3, 3]
    assert (
        tmp_path
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "t1_ref_interrupted"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    # check preprocessing
    dataset.config.datatype = PET_DATATYPE
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(conversion_name="t1_ref_interrupted")

    # check masks
    dataset.config.datatype = T1_DATATYPE
    dataset.individual_masks = [Mask("seg")]
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(conversion_name="t1_ref_interrupted")

    dataset.individual_masks = [Mask("seg"), Mask("brain")]
    dataset.common_masks = [
        Mask(tmp_path / "masks" / "leftHippocampus.nii.gz"),
        Mask(tmp_path / "masks" / "leftHemisphere.nii"),
    ]
    converter = TensorConversion(dataset)
    converter.to_tensors(conversion_name="t1_masks_interrupted")
    with open(tmp_path / "tensor_conversion" / "t1_masks_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["common_masks"]) == sorted(
        ["leftHippocampus.nii.gz", "rightHippocampus.nii.gz", "leftHemisphere.nii"]
    )
    assert (
        tmp_path / "masks" / "tensors" / "t1_masks_interrupted" / "leftHemisphere.pt"
    ).is_file()

    # check also
    dataset.individual_masks = []
    dataset.common_masks = []
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(
            conversion_name="t1_custom_interrupted",
            save_transforms=True,
            check_transforms=False,
        )

    dataset.config.transforms = Transforms(image_transforms=[CustomTransformBis()])
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(
            conversion_name="t1_custom_interrupted",
            save_transforms=True,
            check_transforms=False,
        )

    dataset.config.transforms = Transforms(image_transforms=[CustomTransform()])
    converter = TensorConversion(dataset)
    converter.to_tensors(
        conversion_name="t1_custom_interrupted",
        save_transforms=True,
        check_transforms=False,
    )

    # check transforms
    dataset.config.transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[
            RescaleIntensityConfig(),
            ClampConfig(out_min=-10, out_max=10),
        ],
        sample_transforms=[CropConfig(cropping=1)],
        augmentations=[PadConfig(padding=1)],
    )
    converter = TensorConversion(dataset)
    converter.to_tensors(
        conversion_name="t1_transform_interrupted",
        save_transforms=True,
    )

    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(conversion_name="t1_ref_interrupted", save_transforms=True)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(
            conversion_name="t1_transform_interrupted",
            save_transforms=False,
        )

    dataset.config.transforms = Transforms()
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.to_tensors(
            conversion_name="t1_transform_interrupted", save_transforms=True
        )

    # spacing and shape
    dataset.sub_ses = [
        ("sub-010", "ses-M003"),
        ("sub-010", "ses-M012"),
    ]
    converter = TensorConversion(dataset)
    with pytest.raises(
        TensorConversionError, match="An error occurred during conversion.*"
    ):
        converter.to_tensors(conversion_name="t1_ref_interrupted")
    with pytest.warns(match="Different image shapes found in the dataset:*"):
        converter.to_tensors(conversion_name="t1_ref_interrupted", ignore_spacing=True)
    with open(tmp_path / "tensor_conversion" / "t1_ref_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
            ["sub-010", "ses-M012"],
        ]
    )
    assert conversion_info["spacing"] is None
    assert conversion_info["shape"] is None

    # without shape warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.to_tensors(
            conversion_name="t1_ref_interrupted",
            ignore_spacing=True,
            shape_warning=False,
        )
