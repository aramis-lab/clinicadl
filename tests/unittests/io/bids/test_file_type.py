import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.io import BidsFileType, DwiDti, FlairLinear, PetLinear, T1Linear


class TestBidsFileType:
    @pytest.mark.parametrize(
        "path,match",
        [
            ("anat/sub-000_ses-M000_pet.nii.gz", True),
            (Path("anat/sub-000_ses-M000_trc-abc_pet.nii"), True),
            ("anato/sub-000_ses-M000_pet.nii.gz", False),
            ("anat/sub-000_ses-M000_petscan.nii.gz", False),
            ("anat/sub-000_ses-M000_pet.dicom", False),
            ("sub-000_ses-M000_trc-abc_pet.nii.gz", False),
            ("anat/sub-000_pet.nii.gz", False),
            ("anat/ses-M000_pet.nii.gz", False),
            ("anat/sub-000_ses-M0001_pet.nii.gz", False),
            ("anat/sub-0001_ses-M000_pet.nii.gz", False),
        ],
    )
    def test_1(self, path, match):
        file_type = BidsFileType(datatype="anat", suffix="pet")
        assert file_type.match(path, participant="sub-000", session="ses-M000") == match

    @pytest.mark.parametrize(
        "path,match",
        [
            ("xabcx/sub-000_ses-M000_trc-FDG_res-1x2x2_xabcx.nii.gz", True),
            ("xabcx/sub-000_ses-M000_trc-FDG_res-1x2x2_space-MNI_xabcx.nii.gz", True),
            ("xabcx/sub-000_ses-M000_trc-FDG_res-2x2x2_xabcx.nii.gz", False),
            ("xabcx/sub-000_ses-M000_trc-FDG18_res-1x2x2_xabcx.nii.gz", False),
            ("xabcx/sub-000_ses-M000_trc-FDG_res-1x2x2_abxc.nii.gz", False),
            ("abxc/sub-000_ses-M000_trc-FDG_res-1x2x2_xabcx.nii.gz", False),
            ("xabcx/sub-000_ses-M000_trc-FDG_res-1x2x2_xabcx.nii", False),
        ],
    )
    def test_2(self, path, match):
        file_type = BidsFileType(
            datatype=".*abc.*",
            suffix=".*abc.*",
            extension=".nii.gz",
            with_entities={"trc": "FDG", "res": "1x.*"},
        )
        assert file_type.match(path, participant="sub-000", session="ses-M000") == match

    @pytest.mark.parametrize(
        "path,match",
        [
            ("mri/anat/sub-000_ses-M000_trc-FDG_pet.nii.gz", True),
            ("mri/anat/sub-000_ses-M000_trc-FDG_space-MNI_pet.nii.gz", True),
            ("mri/anat/sub-000_ses-M000_res-2x2x2_run-2_trc-FDG_pet.nii.gz", True),
            ("anat/sub-000_ses-M000_trc-FDG_pet.nii.gz", False),
            ("mri/anat/sub-000_ses-M000_trc-FDG_res-1x2x2_pet.nii.gz", False),
            ("mri/anat/sub-000_ses-M000_trc-FDG_res-2x2x2_run-1_pet.nii.gz", False),
        ],
    )
    def test_3(self, path, match):
        file_type = BidsFileType(
            datatype="mri/anat",
            suffix="pet",
            extension=".nii.gz",
            with_entities={"trc": "FDG"},
            without_entities={"res": "1x.*", "run": "1"},
        )
        assert file_type.match(path, participant="sub-000", session="ses-M000") == match

    @pytest.mark.parametrize(
        "file_type",
        [
            BidsFileType(
                datatype="anat.*",
                suffix="pet.*",
                extension=".nii.*",
                with_entities={"trc": "FDG", "space": "MNI.*"},
                without_entities={"res": "1x.*", "run": "1"},
                description="abc",
            ),
            BidsFileType(
                datatype="anat",
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
            BidsFileType(**parameters, datatype="anat", suffix="pet", extension=".nii")


class TestClinicaPipelines:
    def test_flair(self):
        flair_data = FlairLinear(use_uncropped_image=True)
        assert flair_data.datatype == re.compile("flair_linear")
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii",
            participant="sub-000",
            session="ses-M000",
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not flair_data.match(
            Path("flair_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii",
            participant="sub-000",
            session="ses-M000",
        )

    def test_t1(self):
        t1w_data = T1Linear()
        assert t1w_data.datatype == re.compile("t1_linear")
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii",
            participant="sub-000",
            session="ses-M000",
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not t1w_data.match(
            Path("t1_linear")
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii",
            participant="sub-000",
            session="ses-M000",
        )

    def test_pet(self):
        pet_data = PetLinear(tracer="18FFDG", suvr_reference_region="cerebellumPons2")
        assert pet_data.datatype == re.compile("pet_linear")
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii",
            participant="sub-000",
            session="ses-M000",
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
            participant="sub-000",
            session="ses-M000",
        )
        assert not pet_data.match(
            Path("pet_linear")
            / "sub-000_ses-M000_trc-18FFDG_rec-nacstat_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii",
            participant="sub-000",
            session="ses-M000",
        )
        assert pet_data.description == (
            "PET images with tracer '18FFDG' and reconstruction method 'nacstat', registered to MNI152NLin2009cSym space "
            "using Clinica's 'pet-linear' pipeline with SUVR reference region 'pons2'."
        )

    def test_dwi(self):
        dwi_data = DwiDti(measure="FA", space="normalized")
        assert dwi_data.datatype == re.compile(
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
            participant="sub-000",
            session="ses-M000",
        )
        assert (
            dwi_data.description
            == "DTI FA images in normalized space, preprocessed with Clinica's 'dwi-dti' pipeline."
        )

        dwi_data = DwiDti(measure="MD", space="native")
        assert dwi_data.datatype == re.compile("dwi/dti_based_processing/native_space")
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
            participant="sub-000",
            session="ses-M000",
        )
        assert dwi_data.match(
            os.path.join(
                "dwi",
                "dti_based_processing",
                "native_space",
                "sub-000_ses-M000_space-T1w_MD.nii",
            ),
            participant="sub-000",
            session="ses-M000",
        )
        assert (
            dwi_data.description
            == "DTI MD images in native space, preprocessed with Clinica's 'dwi-dti' pipeline."
        )
