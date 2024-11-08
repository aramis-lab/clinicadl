import json
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.dataset.config.extraction import (
    ALL_EXTRACTION_TYPES,
    ExtractionConfig,
    ExtractionImageConfig,
)
from clinicadl.dataset.config.preprocessing import (
    ALL_PREPROCESSING_TYPES,
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    PETPreprocessingConfig,
    PreprocessingConfig,
)
from clinicadl.dataset.config.utils import (
    get_infos_from_json,
    get_preprocessing,
)
from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.datasets.concat import ConcatDataset
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import (
    check_caps_folder,
    clinicadl_file_reader,
    container_from_filename,
    create_subs_sess_list,
    determine_caps_or_bids,
    get_subject_session_list,
)
from clinicadl.utils.iotools.utils import path_encoder

from .reader import Reader

logger = getLogger("clinicadl.caps_reader")


class CapsReader(Reader):
    def __init__(
        self,
        caps_directory: Path,
        # manager: Optional[ExperimentManager], # I don't think we can give the manager as arg of the class constructor
        from_bids: Optional[Path] = None,
    ):
        """CAPS reader class for handling single-cohort CAPS directories.

        Args:
            caps_directory (Path): Path to the CAPS directory.
            from_bids (Optional[Path], optional): Path to BIDS directory, if applicable. Defaults to None.
        """

        self._get_input_directory(caps_directory, from_bids)

    def tensor_dir(self, file, preprocessing: PreprocessingConfig) -> Path:
        return (
            self.input_directory
            / container_from_filename(file)
            / "deeplearning_prepare_data"
            / "image_based"
            / preprocessing.compute_folder(self.bids)
        )

    def create_caps_json(self):
        """TODO: COMPLETE this method so that it writes all the info needed in a caps.json file"""
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
        """Set the input directory as either BIDS or CAPS.

        Args:
            caps_directory (Path): CAPS directory path.
            from_bids (Optional[Path]): BIDS directory path.
        """
        if from_bids is not None:
            if from_bids.exists():
                self.input_directory = from_bids
                self.bids = True
            else:
                raise ClinicaDLArgumentError("Specified BIDS directory does not exist.")
        else:
            self.input_directory = caps_directory
            check_caps_folder(caps_directory)
            self.bids = False

    def get_input_files(
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

        input_files = self.get_input_files(preprocessing, data_tsv=data_tsv)

        def prepare_image(file: Path):
            output_file_dir = self.tensor_dir(file, preprocessing=preprocessing)

            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / file.name.replace(".nii.gz", ".pt")

            logger.debug(f"Processing of {file}.")
            image_array = nib.loadsave.load(file).get_fdata(dtype="float32")  # type: ignore

            # get some important infos about the image
            info_df = pd.DataFrame(
                [
                    {
                        "mean": image_array.mean(),
                        "std": image_array.std(),
                        "max": image_array.max(),
                        "min": image_array.min(),
                    }
                ]
            )
            info_df.to_csv(
                container_from_filename(file) / "image_info.tsv", sep="\t", index=False
            )

            # extract and save the image tensor
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
            save_tensor(image_tensor.clone(), output_file)
            logger.debug(f"Output tensor saved at {output_file}")

        Parallel(n_jobs=n_proc)(delayed(prepare_image)(file) for file in input_files)

    def write_output_imgs(
        self,
        output_mode: list,
        file: Path,
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
                / "image_based"  # always image as we remove save features option for ROI, SLice and Patch ?
                / mod_subfolder
            )
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / filename
            save_tensor(tensor, output_file)
            logger.debug(f"Output tensor saved at {output_file}")

    def write_preprocessing(
        self,
        preprocessing: PreprocessingConfig,
        extraction: ExtractionConfig,  # I think we need to add transforms and now extraction is inside Transforms ?
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
        preprocessing, _, transforms = self.get_infos_from_json(
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
            transforms = Transforms(
                extraction=ExtractionImageConfig()
            )  # means no transforms and image (default)

        return CapsDataset(
            caps_directory=self.input_directory,
            preprocessing=preprocessing,
            data_df=data_df,
            transforms=transforms,
        )

    def load_data_test(self, test_path: Path, baseline=True):
        """
        Load data not managed by split_manager.

        Args:
            test_path (str): path to the test TSV files / split directory / TSV file for multi-cohort
            baseline (bool): If True baseline sessions only used (split_dir handling only).
        """
        # TODO: computes baseline sessions on-the-fly to manager TSV file case

        if test_path.suffix != ".tsv" or not test_path.is_file():
            raise ClinicaDLConfigurationError(
                "Test path should be a TSV file. Please provide a valid TSV file path."
            )
        tsv_df = pd.read_csv(test_path, sep="\t")
        multi_col = {"cohort", "path"}
        if multi_col.issubset(tsv_df.columns.values):
            raise ClinicaDLConfigurationError(
                "To use multi-cohort framework, please add 'multi_cohort=true' in your configuration file or '--multi_cohort' flag to the command line."
            )
        test_path = self.check_test_path(test_path=test_path, baseline=baseline)
        test_df = pd.read_csv(test_path, sep="\t")
        test_df.reset_index(inplace=True, drop=True)
        test_df["cohort"] = "single"

        return test_df
