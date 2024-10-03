import abc
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from clinicadl.caps_dataset.data_config import DataConfig
from clinicadl.splitter.validation import ValidationConfig
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.clinica_utils import check_caps_folder

logger = getLogger("clinicadl.split_manager")


def init_split_manager(
    parameters,
    split_list=None,
):
    data_config = DataConfig(**parameters)
    split_config = ValidationConfig(**parameters)

    return Splitter(
        data_config=data_config, validation_config=split_config, split_list=split_list
    )


class Splitter:
    def __init__(
        self,
        data_config: DataConfig,
        validation_config: ValidationConfig,
        split_list: Optional[List[int]] = None,
    ):
        """_summary_

        Parameters
        ----------
        data_config : DataConfig
            _description_
        validation_config : ValidationConfig
            _description_
        split_list : Optional[List[int]] (optional, default=None)
            _description_

        Examples
        --------
        >>> _input_
        _output_

        Notes
        -----
        _notes_

        See Also
        --------
        - _related_
        """
        self.data_config = data_config
        self.validation_config = validation_config
        self.split_list = split_list

        self.caps_dict = self.data_config.caps_dict  # TODO : check if useful ?

    def max_length(self) -> int:
        """Maximum number of splits"""
        return self.validation_config.n_splits

    def __len__(self):
        if not self.split_list:
            return self.validation_config.n_splits
        else:
            return len(self.split_list)

    @property
    def allowed_splits_list(self):
        """
        List of possible splits if no restriction was applied

        Returns:
            list[int]: list of all possible splits
        """
        return [i for i in range(self.validation_config.n_splits)]

    def __getitem__(self, item) -> Dict:
        """
        Returns a dictionary of DataFrames with train and validation data.

        Args:
            item (int): Index of the split wanted.
        Returns:
            Dict[str:pd.DataFrame]: dictionary with two keys (train and validation).
        """
        self._check_item(item)

        if self.data_config.multi_cohort:
            tsv_df = pd.read_csv(self.validation_config.tsv_path, sep="\t")
            train_df = pd.DataFrame()
            valid_df = pd.DataFrame()
            found_diagnoses = set()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.at[idx, "cohort"]
                cohort_path = Path(tsv_df.at[idx, "path"])
                cohort_diagnoses = (
                    tsv_df.at[idx, "diagnoses"].replace(" ", "").split(",")
                )
                if bool(set(cohort_diagnoses) & set(self.data_config.diagnoses)):
                    target_diagnoses = list(
                        set(cohort_diagnoses) & set(self.data_config.diagnoses)
                    )

                    cohort_train_df, cohort_valid_df = self.concatenate_diagnoses(
                        item, cohort_path=cohort_path, cohort_diagnoses=target_diagnoses
                    )
                    cohort_train_df["cohort"] = cohort_name
                    cohort_valid_df["cohort"] = cohort_name
                    train_df = pd.concat([train_df, cohort_train_df])
                    valid_df = pd.concat([valid_df, cohort_valid_df])
                    found_diagnoses = found_diagnoses | (
                        set(cohort_diagnoses) & set(self.data_config.diagnoses)
                    )

            if found_diagnoses != set(self.data_config.diagnoses):
                raise ValueError(
                    f"The diagnoses found in the multi cohort dataset {found_diagnoses} "
                    f"do not correspond to the diagnoses wanted {set(self.data_config.diagnoses)}."
                )
            train_df.reset_index(inplace=True, drop=True)
            valid_df.reset_index(inplace=True, drop=True)
        else:
            train_df, valid_df = self.concatenate_diagnoses(item)
            train_df["cohort"] = "single"
            valid_df["cohort"] = "single"

        return {
            "train": train_df,
            "validation": valid_df,
        }

    @staticmethod
    def get_dataframe_from_tsv_path(tsv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(tsv_path, sep="\t")
        list_columns = df.columns.values

        if (
            "diagnosis" not in list_columns
            # or "age" not in list_columns
            # or "sex" not in list_columns
        ):
            parents_path = tsv_path.resolve().parent
            labels_path = parents_path / "labels.tsv"
            while (
                not labels_path.is_file()
                and ((parents_path / "kfold.json").is_file())
                or (parents_path / "split.json").is_file()
            ):
                parents_path = parents_path.parent
            try:
                labels_df = pd.read_csv(labels_path, sep="\t")
                df = pd.merge(
                    df,
                    labels_df,
                    how="inner",
                    on=["participant_id", "session_id"],
                )
            except Exception:
                pass
        return df

    @staticmethod
    def load_data(
        tsv_path: Path, cohort_diagnoses: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = Splitter.get_dataframe_from_tsv_path(tsv_path)
        df = df[df.diagnosis.isin((cohort_diagnoses))]
        df.reset_index(inplace=True, drop=True)
        return df

    def concatenate_diagnoses(
        self,
        split,
        cohort_path: Optional[Path] = None,
        cohort_diagnoses: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Concatenated the diagnoses needed to form the train and validation sets."""

        if cohort_diagnoses is None:
            cohort_diagnoses = self.data_config.diagnoses

        tmp_cohort_path = (
            cohort_path if cohort_path is not None else self.validation_config.tsv_path
        )
        train_path, valid_path = self._get_tsv_paths(
            tmp_cohort_path,
            split,
        )

        logger.debug(f"Training data loaded at {train_path}")
        if self.data_config.baseline:
            train_path = train_path / "train_baseline.tsv"
        else:
            train_path = train_path / "train.tsv"
        train_df = self.load_data(train_path, cohort_diagnoses)

        logger.debug(f"Validation data loaded at {valid_path}")
        if self.validation_config.valid_longitudinal:
            valid_path = valid_path / "validation.tsv"
        else:
            valid_path = valid_path / "validation_baseline.tsv"
        valid_df = self.load_data(valid_path, cohort_diagnoses)

        return train_df, valid_df

    @abc.abstractmethod
    def _get_tsv_paths(self, cohort_path, *args) -> Tuple[Path, Path]:
        """
        Computes the paths to the TSV files needed depending on the split structure.

        Args:
            cohort_path (str): path to the split structure of a cohort.
            split (int): Index of the split.
        Returns:
            train_path (str): path to the directory containing training data.
            valid_path (str): path to the directory containing validation data.
        """
        if args is not None:
            for split in args:
                train_path = cohort_path / f"split-{split}"
                valid_path = cohort_path / f"split-{split}"
            return train_path, valid_path
        else:
            train_path = cohort_path
            valid_path = cohort_path
            return train_path, valid_path

    @abc.abstractmethod
    def split_iterator(self):
        """Returns an iterable to iterate on all splits wanted."""
        if not self.split_list:
            return range(self.validation_config.n_splits)
        else:
            return self.split_list

    def _check_item(self, item):
        if item not in self.allowed_splits_list:
            raise IndexError(
                f"Split index {item} out of allowed splits {self.allowed_splits_list}."
            )
