import json
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.dataset.config.preprocessing import (
    ALL_PREPROCESSING_TYPES,
    PreprocessingConfig,
)
from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.datasets.concat import ConcatDataset
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError

from .caps_reader import CapsReader
from .reader import Reader


class CapsMultiReader(Reader):
    def __init__(
        self,
        caps_directory: Path,
        # manager: Optional[ExperimentManager],
        from_bids: Optional[Path] = None,
    ):
        """CAPS reader for handling multi-cohort CAPS directories."""
        self.caps_dict = {}

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
            cohort_name = caps_df.at[idx, "cohort"]
            cohort_path = Path(caps_df.at[idx, "path"])

            caps_reader = CapsReader(caps_directory=cohort_path)
            self.caps_dict[cohort_name] = caps_reader

        self.input_directory = caps_reader.input_directory
        self.bids = caps_reader.bids

    def get_dataset(
        self,
        preprocessing: PreprocessingConfig,
        sub_ses_tsv: Optional[Path] = None,
        transforms: Optional[Transforms] = None,
    ) -> ConcatDataset:
        dataset_list = []
        for cohort, caps_reader in self.caps_dict:
            dataset = caps_reader.get_dataset(
                preprocessing=preprocessing,
                sub_ses_tsv=sub_ses_tsv,
                transforms=transforms,
            )
            dataset_list.append(dataset)

        return ConcatDataset(dataset_list)

    def load_data_test(self, test_path: Path, baseline=True):
        if test_path.suffix != ".tsv":
            raise ClinicaDLArgumentError(
                "If multi_cohort is given, the TSV_DIRECTORY argument should be a path to a TSV file."
            )
        else:
            tsv_df = pd.read_csv(test_path, sep="\t")
            mandatory_col = ("cohort", "path")
            if not set(mandatory_col).issubset(tsv_df.columns.values):
                raise ClinicaDLTSVError(
                    f"Columns of the TSV file must include {mandatory_col}"
                )
            test_df = pd.DataFrame()
            for idx in range(len(tsv_df)):
                cohort_path = Path(tsv_df.at[idx, "path"])
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_test_df = self.caps_dict[cohort_name].load_data_test(
                    cohort_path, baseline=baseline
                )
                cohort_test_df["cohort"] = cohort_name
                test_df = pd.concat([test_df, cohort_test_df])
            test_df.reset_index(inplace=True, drop=True)
            return test_df
