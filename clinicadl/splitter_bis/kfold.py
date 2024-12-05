import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, NonNegativeInt, PositiveInt

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.utils import tsv_to_df
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.splitter.split_output import Split
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal
from clinicadl.utils.exceptions import ClinicaDLTSVError

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class KFoldConfig(BaseModel):
    subset_name: str = "validation"
    n_splits: PositiveInt = 5
    stratification: Optional[List[str]] = None
    ignore_demographics: bool = False
    valid_longitudinal: bool = False


class DataLoaderConfig(BaseModel):
    n_procs: PositiveInt = 3
    batch_size: PositiveInt = 10


class KFolder:
    def __init__(
        self, split_dir: Path, caps_dataset: CapsDataset, manager: ExperimentManager
    ) -> None:
        self.dataset = caps_dataset
        self.reader = caps_dataset.caps_reader
        self.df = caps_dataset.df
        self.manager = manager
        self.split_dir = split_dir

        self.config = self._get_config_from_json()

    def _get_config_from_json(self):
        kfold_json_path = self.split_dir / "kfold.json"

        if not kfold_json_path.exists():
            raise ClinicaDLTSVError(f"No kfold.json file found in {self.split_dir}")

        with kfold_json_path.open(mode="r") as f:
            preprocessing_dict = json.load(f)

        return KFoldConfig(**preprocessing_dict)

    def _get_split_path(self, split_number: PositiveInt):
        return self.split_dir / f"split-{split_number}"

    def _get_tsv_path(
        self, split_number: PositiveInt, baseline: bool, subset_name: str
    ):
        if baseline:
            filename = f"{subset_name}_baseline.tsv"
        else:
            filename = f"{subset_name}.tsv"

        tsv_path = self._get_split_path(split_number) / filename

        if not tsv_path.exists():
            raise ClinicaDLTSVError(f"No {tsv_path} file found")

        return tsv_path

    def _get_train_tsv_path(self, split_number: PositiveInt, baseline: bool = False):
        return self._get_tsv_path(
            split_number=split_number, baseline=baseline, subset_name="train"
        )

    def _get_test_tsv_path(
        self,
        split_number: PositiveInt,
        subset_name: str = "validation",
        baseline: bool = True,
    ):
        return self._get_tsv_path(
            split_number=split_number, baseline=baseline, subset_name=subset_name
        )

    def _check_split_and_dataset(self):
        tsv_path = self._get_test_tsv_path(0)
        train_path = self._get_train_tsv_path(0)
        df_split = pd.concat(
            [pd.read_csv(tsv_path, sep="\t"), pd.read_csv(train_path, sep="\t")]
        )

        common_rows = pd.merge(df_split, self.df, how="inner")
        all_included = len(common_rows) == len(df_split)

        if not all_included:
            missing_rows = pd.concat(
                [df_split, self.df], ignore_index=True
            ).drop_duplicates(keep=False)

            err_message = "Missing rows: \n"
            for row in missing_rows:
                err_message += f" - {row} \n"

            raise ClinicaDLTSVError(
                "Some couples (participanst_id, session_id) are not in the dataset,",
                err_message,
            )

    def get_splits(
        self,
        splits: Optional[List[int]] = None,
        dataloader_config: Optional[DataLoaderConfig] = None,
    ):
        pass
