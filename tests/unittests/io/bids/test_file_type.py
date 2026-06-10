import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.io.bids import (
    BidsFileType,
    DwiDti,
    FlairLinear,
    PetLinear,
    T1Linear,
    TensorType,
)
from clinicadl.utils.bids import BidsFile


class SubFileType(BidsFileType):
    pass


def test_equal():
    file_type = SubFileType(data_type="abc", suffix="abc", description="a desc")
    other = SubFileType(data_type="abc", suffix="abc", description="another desc")
    assert file_type == other
    other.suffix = "bcd"
    assert file_type != other


class TestBidsFileType:
    @pytest.mark.parametrize(
        "path,match",
        [
            ("sub-000_ses-M000_pet.nii.gz", True),
            (Path("sub-000_ses-M000_trc-abc_pet.nii"), True),
            ("pet/sub-000_ses-M000_pet.nii.gz", False),
            ("sub-000_ses-M000_petscan.nii.gz", False),
            ("sub-000_ses-M000_pet.dicom", False),
            ("sub-000_pet.nii.gz", False),
            ("ses-M000_pet.nii.gz", False),
            ("sub-000_ses-M001_pet.nii.gz", False),
            ("sub-001_ses-M000_pet.nii.gz", False),
        ],
    )
    def test_1(self, path, match):
        file_type = BidsFileType(suffix="pet")
        assert (
            file_type.match(path, participant_id="sub-000", session_id="ses-M000")
            == match
        )

    @pytest.mark.parametrize(
        "path,match",
        [
            ("xabcx/sub-000_ses-M000_trc-FDG_res-1x2x2_xabcx.nii.gz", True),
            ("xabcx/sub-001_ses-M001_trc-FDG_res-1x2x2_space-MNI_xabcx.nii.gz", True),
            ("abc/trc-FDG_res-1x2x2_space-MNI_abc.nii.gz", True),
            ("xabcx/trc-FDG_res-2x2x2_xabcx.nii.gz", False),
            ("xabcx/trc-FDG18_res-1x2x2_xabcx.nii.gz", False),
            ("xabcx/trc-FDG_res-1x2x2_abxc.nii.gz", False),
            ("abxc/trc-FDG_res-1x2x2_xabcx.nii.gz", False),
            ("xabcx/trc-FDG_res-1x2x2_xabcx.nii", False),
        ],
    )
    def test_2(self, path, match):
        file_type = BidsFileType(
            data_type=r".*abc.*",
            suffix=r".*abc.*",
            extension=".nii.gz",
            with_entities={"trc": "FDG", "res": "1x.*"},
        )
        assert file_type.match(path) == match

    @pytest.mark.parametrize(
        "path,match",
        [
            ("mri/anat/sub-000_trc-18FDG_pet.nii.gz", True),
            ("mri/anat/sub-000_ses-M001_trc-FDG_space-MNI_pet.nii.gz", True),
            ("mri/anat/sub-000_ses-M000_res-2x2x2_run-2_trc-FDG_pet.nii.gz", True),
            ("mri/anat/sub-000_trc-19FDG_pet.nii.gz", False),
            ("anat/sub-000_trc-FDG_pet.nii.gz", False),
            ("mri/anat/sub-000_trc-FDG_res-1x2x2_pet.nii.gz", False),
            ("mri/anat/sub-000_trc-FDG_res-2x2x2_run-1_pet.nii.gz", False),
        ],
    )
    def test_3(self, path, match):
        file_type = BidsFileType(
            data_type="mri/anat",
            suffix="pet",
            extension=".nii.gz",
            with_entities={"trc": ".*FDG"},
            without_entities={"res": "1x.*", "run": "1", "trc": "19FDG"},
        )
        assert file_type.match(path, participant_id="sub-000") == match

    @pytest.mark.parametrize(
        "path,match",
        [
            ("space-MNI152NLin2009cSym_res-1x1x1_sessions.tsv", True),
            ("sessions.tsv", False),
            ("sub-000/space-MNI152NLin2009cSym_res-1x1x1_sessions.tsv", False),
        ],
    )
    def test_4(self, path, match):
        file_type = BidsFileType(
            suffix="sessions",
            extension=".tsv",
            with_entities={"space": "MNI152.*", "res": "1x1x1"},
        )
        assert file_type.match(path) == match

    @pytest.mark.parametrize(
        "file_type",
        [
            BidsFileType(
                data_type="anat.*",
                suffix="pet.*",
                extension=".nii.*",
                with_entities={"trc": "FDG", "space": "MNI.*"},
                without_entities={"res": "1x.*", "run": "1"},
                description="abc",
            ),
            BidsFileType(
                data_type="anat",
                suffix="pet",
                extension=".nii",
            ),
        ],
    )
    def test_serialize_deserialize(self, file_type, tmp_path):
        file_type.to_json(tmp_path / "file_type.json")

        new_filetype = file_type.from_json(tmp_path / "file_type.json")
        assert new_filetype == file_type

    @pytest.mark.parametrize(
        "parameters",
        [
            {"with_entities": {"abc_": "abc", "space": "MNI"}},
            {"without_entities": {"abc ": "abc", "space": "MNI"}},
        ],
    )
    def test_bad_inputs(self, parameters):
        with pytest.raises(ValidationError):
            BidsFileType(**parameters, data_type="anat", suffix="pet", extension=".nii")


class TestClinicaPipelines:
    def test_flair(self):
        flair_data = FlairLinear(use_uncropped_image=True)
        assert flair_data.data_type == re.compile("flair_linear")
        assert flair_data.suffix == re.compile("FLAIR")
        assert flair_data.extension == re.compile(".nii.*")
        assert flair_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
        }
        assert flair_data.without_entities == {"desc": re.compile("Crop")}
        assert flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert (
            flair_data.description
            == "FLAIR images registered to MNI152NLin2009cSym space using Clinica's 'flair-linear' pipeline."
        )

        flair_data = FlairLinear()
        assert flair_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
            "desc": re.compile("Crop"),
        }
        assert not flair_data.without_entities
        assert flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )

    def test_t1(self):
        t1w_data = T1Linear()
        assert t1w_data.data_type == re.compile("t1_linear")
        assert t1w_data.suffix == re.compile("T1w")
        assert t1w_data.extension == re.compile(".nii.*")
        assert t1w_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
            "desc": re.compile("Crop"),
        }
        assert not t1w_data.without_entities
        assert t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert t1w_data.description == (
            "T1 weighted images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline, "
            "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)."
        )

        t1w_data = T1Linear(use_uncropped_image=True)
        assert t1w_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
        }
        assert t1w_data.without_entities == {"desc": re.compile("Crop")}
        assert t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )

    def test_pet(self):
        pet_data = PetLinear(tracer="18FFDG", suvr_reference_region="cerebellumPons2")
        assert pet_data.data_type == re.compile("pet_linear")
        assert pet_data.suffix == re.compile("pet")
        assert pet_data.extension == re.compile(".nii.*")
        assert pet_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
            "trc": re.compile("18FFDG"),
            "suvr": re.compile("cerebellumPons2"),
            "desc": re.compile("Crop"),
        }
        assert not pet_data.without_entities
        assert pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert pet_data.description == (
            "PET images with tracer '18FFDG' registered to MNI152NLin2009cSym space using Clinica's "
            "'pet-linear' pipeline with SUVR reference region 'cerebellumPons2', and cropped "
            "(matrix size 169×208×179, 1 mm isotropic voxels)."
        )

        pet_data = PetLinear(
            tracer="18FFDG",
            suvr_reference_region="pons2",
            reconstruction="nacstat",
            use_uncropped_image=True,
        )
        assert pet_data.with_entities == {
            "space": re.compile("MNI152NLin2009cSym"),
            "res": re.compile("1x1x1"),
            "trc": re.compile("18FFDG"),
            "suvr": re.compile("pons2"),
            "rec": re.compile("nacstat"),
        }
        assert pet_data.without_entities == {"desc": re.compile("Crop")}
        assert pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_rec-nacstat_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert not pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_rec-nacstat_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii",
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert pet_data.description == (
            "PET images with tracer '18FFDG' and reconstruction method 'nacstat', registered to MNI152NLin2009cSym space "
            "using Clinica's 'pet-linear' pipeline with SUVR reference region 'pons2'."
        )

    def test_dwi(self):
        dwi_data = DwiDti(measure="FA", space="normalized")
        assert dwi_data.data_type == re.compile(
            "dwi/dti_based_processing/normalized_space"
        )
        assert dwi_data.suffix == re.compile("FA")
        assert dwi_data.extension == re.compile(".nii.*")
        assert dwi_data.with_entities == {
            "space": re.compile("MNI152Lin"),
            "res": re.compile("1x1x1"),
        }
        assert dwi_data.without_entities is None
        assert dwi_data.match(
            os.path.join(
                "dwi",
                "dti_based_processing",
                "normalized_space",
                "sub-000_ses-M000_space-MNI152Lin_res-1x1x1_FA.nii",
            ),
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert (
            dwi_data.description
            == "DTI FA images in normalized space, preprocessed with Clinica's 'dwi-dti' pipeline."
        )

        dwi_data = DwiDti(measure="MD", space="native")
        assert dwi_data.data_type == re.compile("dwi/dti_based_processing/native_space")
        assert dwi_data.with_entities == {
            "space": re.compile(r"\b(b0|T1w)\b"),
        }
        assert dwi_data.match(
            os.path.join(
                "dwi",
                "dti_based_processing",
                "native_space",
                "sub-000_ses-M000_space-b0_MD.nii",
            ),
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert dwi_data.match(
            os.path.join(
                "dwi",
                "dti_based_processing",
                "native_space",
                "sub-000_ses-M000_space-T1w_MD.nii",
            ),
            participant_id="sub-000",
            session_id="ses-M000",
        )
        assert (
            dwi_data.description
            == "DTI MD images in native space, preprocessed with Clinica's 'dwi-dti' pipeline."
        )


class TestTensorType:
    def test_init(self):
        tensor = TensorType(entities={"conv": "abc", "trc": r"18FD.*"})
        assert tensor.with_entities == {
            "conv": re.compile("abc"),
            "trc": re.compile(r"18FD.*"),
        }
        assert tensor.extension == re.compile(".pt")
        assert tensor.suffix == re.compile("tensors")
        assert tensor.data_type == re.compile("tensors")
        assert tensor.without_entities is None
        assert tensor.description == "Outputs of the tensor conversion."

    @pytest.mark.parametrize(
        "image,individual_masks,common_masks,transformed,expected",
        [
            (
                BidsFileType(
                    data_type="",
                    suffix="pet",
                    with_entities={
                        "trc": "18FDG",
                        "res": "1x1x1",
                        "space": r"MNI.*",
                        "desc": "Crop",
                        "run": "1",
                    },
                ),
                [
                    BidsFileType(
                        data_type="",
                        suffix="",
                        with_entities={
                            "trc": "18FDG",
                            "res": "1x1x1",
                            "space": r"MNI.*",
                            "desc": "Crop",
                        },
                    ),
                    BidsFileType(
                        data_type="",
                        suffix="",
                        with_entities={
                            "trc": "18FDG",
                            "res": "1x1x1",
                            "space": r"MNI.*",
                            "desc": "Crop",
                            "run": "1",
                        },
                    ),
                ],
                [
                    BidsFile(
                        "abc/sub-000_ses-M000_trc-18FDG_res-2x2x2_space-MNI_desc-Crop_run-1_mask.nii.gz"
                    ),
                    BidsFile(
                        "sub-001_ses-M001_trc-18FDG_res-1x1x1_space-MNI_run-1_mask.nii.gz"
                    ),
                ],
                False,
                {"trc": re.compile("18FDG"), "src": re.compile("pet")},
            ),
            (
                BidsFileType(
                    data_type="",
                    suffix=r"pet*",
                    with_entities={
                        "trc": "18FDG",
                        "res": "1x1x1",
                    },
                ),
                [],
                [],
                False,
                {
                    "trc": re.compile("18FDG"),
                    "res": re.compile("1x1x1"),
                },
            ),
            (
                BidsFileType(
                    data_type="",
                    suffix="T1w",
                    with_entities={
                        "res": "1x1x1",
                    },
                ),
                [],
                [],
                True,
                {"src": re.compile("T1w")},
            ),
            (
                BidsFileType(
                    data_type="",
                    suffix="pet",
                ),
                [
                    BidsFileType(
                        data_type="",
                        suffix="mask",
                    ),
                ],
                [
                    BidsFile("abc/mask.nii.gz"),
                ],
                False,
                {"src": re.compile("pet")},
            ),
        ],
    )
    def test_from_source_file_types(
        self, image, individual_masks, common_masks, transformed, expected
    ):
        tensor = TensorType.from_source_file_types(
            conversion_name="abc",
            image=image,
            transformed=transformed,
            individual_masks=individual_masks,
            common_masks=common_masks,
        )
        expected["conv"] = re.compile("abc")
        assert tensor.with_entities == expected
