import json
from abc import abstractmethod
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.dataset.concat import ConcatDataset
from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    PreprocessingConfig,
    T1PreprocessingConfig,
    T2PreprocessingConfig,
)
from clinicadl.dataset.config.utils import (
    get_preprocessing,
    get_preprocessing_and_mode_from_json,
)
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SubFolder,
    Suffix,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.iotools.clinica_utils import (
    check_caps_folder,
    clinicadl_file_reader,
    container_from_filename,
    create_subs_sess_list,
    determine_caps_or_bids,
    get_subject_session_list,
)
from clinicadl.utils.iotools.utils import path_encoder

logger = getLogger("clinicadl.caps_reader")


class Reader:
    def __init__(self, input_dir: Path, bids: bool) -> None:
        self.input_directory = input_dir
        self.bids = bids

    def get_preprocessing(
        self, preprocessing: Union[str, Preprocessing]
    ) -> PreprocessingConfig:
        """TO COMPLETE"""

        preprocessing_ = Preprocessing(preprocessing)
        print(preprocessing_)
        subjects, sessions = get_subject_session_list(
            input_dir=self.input_directory, is_bids_dir=self.bids
        )
        if (
            self.input_directory
            / "subjects"
            / subjects[0]
            / sessions[0]
            / (preprocessing_.value).replace("-", "_")
        ).is_dir():
            preprocessing_config = get_preprocessing(preprocessing_)()
            preprocessing_config.from_bids = self.bids
            pattern = preprocessing_config.file_type.pattern

            def get_value(enum, pattern: str):
                for value in enum:
                    if value.value in pattern:
                        return value
                raise ValueError(
                    f"We can't find a value matching {[e.value for e in enum]} in the pattern {pattern}"
                )

            if isinstance(preprocessing_config, PETPreprocessingConfig):
                preprocessing_config.tracer = get_value(Tracer, pattern)
                preprocessing_config.suvr_reference_region = get_value(
                    SUVRReferenceRegions, pattern
                )

            elif isinstance(preprocessing_config, DTIPreprocessingConfig):
                preprocessing_config.dti_measure = get_value(DTIMeasure, pattern)
                preprocessing_config.dti_space = get_value(DTISpace, pattern)

            elif isinstance(preprocessing_config, CustomPreprocessingConfig):
                # TODO: add something to find the custom pattern
                pass
        else:
            raise FileNotFoundError(
                f"The preprocessing folder {preprocessing} does not exist."
            )
        return preprocessing_config

    def get_preprocessing_and_extraction_from_json(self, preprocessing_json: Path):
        if not preprocessing_json.is_file():
            raise FileNotFoundError(
                f"The provided preprocessing JSON file {preprocessing_json} does not exist."
            )

        preprocessing, extraction = get_preprocessing_and_mode_from_json(
            preprocessing_json
        )
        return preprocessing, extraction


class CapsReader(Reader):
    def __init__(
        self,
        caps_directory: Path,
        # manager: Optional[ExperimentManager],
        from_bids: Optional[Path] = None,
    ):
        """TO COMPLETE"""

        # self.manager = manager
        self._get_input_directory(caps_directory, from_bids)

    def create_caps_json(self):
        caps_json = self.input_directory / "caps.json"
        if caps_json.is_file:
            with open(caps_json, "a") as f:
                caps_data = json.load(f)
                return caps_data

        else:
            with open(caps_json, "w") as f:
                f.write("tests")
                caps_data = json.load(f)
                return caps_data

    def _get_input_directory(
        self, caps_directory: Path, from_bids: Optional[Path] = None
    ):
        # Get subject and session list
        if from_bids is not None:
            try:
                self.input_directory = Path(from_bids)
            except ClinicaDLArgumentError:
                logger.warning("Your BIDS directory doesn't exist.")
            logger.debug(f"BIDS directory: {self.input_directory}.")
            self.bids = True
        else:
            self.input_directory = caps_directory
            check_caps_folder(self.input_directory)
            logger.debug(f"CAPS directory: {self.input_directory}.")
            self.bids = False

    def prepare_extraction(
        self, preprocessing: PreprocessingConfig, data_tsv: Optional[Path] = None
    ):
        subjects, sessions = get_subject_session_list(
            self.input_directory, data_tsv, self.bids, False, None
        )
        logger.debug(f"List of subjects: \n{subjects}.")
        logger.debug(f"List of sessions: \n{sessions}.")

        file_type = preprocessing.get_filetype()

        input_files = clinicadl_file_reader(
            subjects, sessions, self.input_directory, file_type.model_dump()
        )[0]
        logger.debug(f"Selected image file name list: {input_files}.")

        return input_files

    def prepare_data(
        self,
        preprocessing: PreprocessingConfig,
        data_tsv: Optional[Path] = None,
        n_proc: int = 2,
        use_uncropped_images: bool = False,
    ):
        """TO COMPLETE"""

        input_files = self.prepare_extraction(preprocessing, data_tsv=data_tsv)

        def prepare_image(file: Path):
            output_file_dir = (
                self.input_directory
                / container_from_filename(file)
                / "deeplearning_prepare_data"
                / SubFolder.IMAGE.value
                / preprocessing.compute_folder(self.bids)
            )

            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / file.name.replace(
                Suffix.NII_GZ.value, Suffix.PT.value
            )

            logger.debug(f"Processing of {file}.")
            image_array = nib.loadsave.load(file).get_fdata(dtype="float32")  # type: ignore

            # get some important infos about the image
            info_df = pd.DataFrame(columns=["mean", "std", "max", "min"])
            info_df.loc[0] = [
                image_array.mean(),
                image_array.std(),
                image_array.max(),
                image_array.min(),
            ]
            info_df.to_csv("image_info.tsv", sep="\t")

            # extract and save the image tensor
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
            save_tensor(image_tensor.clone(), output_file)
            logger.debug(f"Output tensor saved at {output_file}")

        Parallel(n_jobs=n_proc)(delayed(prepare_image)(file) for file in input_files)

    def write_output_imgs(
        self,
        output_mode: list,
        file: Path,
        subfolder: SubFolder,
        preprocessing: PreprocessingConfig,
    ):
        # Write the extracted tensor on a .pt file
        container = container_from_filename(file)
        mod_subfolder = preprocessing.compute_folder(self.bids)

        for filename, tensor in output_mode:
            output_file_dir = (
                self.input_directory
                / container
                / "deeplearning_prepare_data"
                / subfolder.value
                / mod_subfolder
            )
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / filename
            save_tensor(tensor, output_file)
            logger.debug(f"Output tensor saved at {output_file}")

    def write_preprocessing(
        self, preprocessing: PreprocessingConfig, extraction: ExtractionConfig
    ) -> Path:
        extract_dir = self.input_directory / "tensor_extraction"
        extract_dir.mkdir(parents=True, exist_ok=True)

        json_path = extract_dir / extraction.extract_json

        if json_path.is_file():
            raise FileExistsError(
                f"JSON file at {json_path} already exists. "
                f"Please choose another name for your preprocessing file."
            )

        preprocessing_dict = preprocessing.model_dump()
        preprocessing_dict.update(extraction.model_dump())

        with json_path.open(mode="w") as json_file:
            json.dump(preprocessing_dict, json_file, default=path_encoder)
        return json_path

    def get_dataset_from_json(
        self, json_path: Path, sub_ses_tsv: Optional[Path] = None
    ):
        preprocessing, transforms = self.get_preprocessing_and_extraction_from_json(
            json_path
        )  # we need to add the transforms infos in the caps.json

        return self.get_dataset(
            preprocessing=preprocessing, transforms=transforms, sub_ses_tsv=sub_ses_tsv
        )

    def get_dataset(
        self,
        preprocessing: PreprocessingConfig,
        sub_ses_tsv: Optional[Path] = None,
        transforms: Optional[Transforms] = None,
    ) -> CapsDataset:
        """TO COMPLETE"""

        if sub_ses_tsv is None:
            sub_ses_tsv = create_subs_sess_list(
                self.input_directory, output_dir=self.input_directory
            )
        elif not sub_ses_tsv.is_file():
            raise FileNotFoundError(
                f"The provided sub_ses_tsv file {sub_ses_tsv} does not exist."
            )

        data_df = pd.read_csv(
            sub_ses_tsv, sep="\t"
        )  # create function to check if we have the part and sess columns and to read the csv

        if transforms is None:
            logger.info(
                "No transforms was provided. We will use the default transforms. Check the documentation for more information"
            )
            transforms = Transforms(extraction=ExtractionImageConfig())

        # if isinstance(transforms.extraction, ExtractionImageConfig):
        #     dataset_type = CapsDatasetImage

        # elif isinstance(transforms.extraction, ExtractionSliceConfig):
        #     dataset_type = CapsDatasetSlice

        # elif isinstance(transforms.extraction, ExtractionPatchConfig):
        #     dataset_type = CapsDatasetPatch

        # elif isinstance(transforms.extraction, ExtractionROIConfig):
        #     dataset_type = CapsDatasetRoi

        # else:
        #     raise NotImplementedError(
        #         f"Mode {transforms.extraction.extract_method.value} is not implemented."
        #     )

        return CapsDataset(
            caps_directory=self.input_directory,
            preprocessing=preprocessing,
            data_df=data_df,
            transforms=transforms,
        )


class CapsMultiReader(Reader):
    def __init__(
        self,
        caps_directory: Path,
        # manager: Optional[ExperimentManager],
        from_bids: Optional[Path] = None,
    ):
        """TO COMPLETE"""
        self.caps_list = []

        if not caps_directory.is_file():
            raise FileNotFoundError(
                f"The provided caps directory {caps_directory} does not exist. Careful: It must be a tsv file in multi-cohort."
            )

        caps_df = pd.read_csv(caps_directory, sep="\t")

        if not set(("cohort", "path")).issubset(caps_df.columns.values):
            raise ClinicaDLTSVError(
                "Columns of the TSV file used for CAPS location must include cohort and path"
            )

        for idx in range(len(caps_df)):
            # cohort_name = caps_df.at[idx, "cohort"]
            cohort_path = Path(caps_df.at[idx, "path"])

            caps_reader = CapsReader(caps_directory=cohort_path)
            self.caps_list.append(caps_reader)

        self.input_directory = caps_reader.input_directory
        self.bids = caps_reader.bids

    def get_dataset(
        self,
        preprocessing: PreprocessingConfig,
        sub_ses_tsv: Optional[Path] = None,
        transforms: Optional[Transforms] = None,
    ) -> ConcatDataset:
        dataset_list = []
        for caps_reader in self.caps_list:
            dataset = caps_reader.get_dataset(
                preprocessing=preprocessing,
                sub_ses_tsv=sub_ses_tsv,
                transforms=transforms,
            )
            dataset_list.append(dataset)

        return ConcatDataset(dataset_list)
