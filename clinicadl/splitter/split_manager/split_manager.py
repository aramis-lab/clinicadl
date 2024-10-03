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
    validation,
    parameters,
    split_list=None,
    ssda_bool: bool = False,
    caps_target: Optional[Path] = None,
    tsv_target_lab: Optional[Path] = None,
):
    print(parameters)
    # split_class = getattr(split_manager, validation)
    # args = list(
    #     split_class.__init__.__code__.co_varnames[
    #         : split_class.__init__.__code__.co_argcount
    #     ]
    # )
    # args.remove("self")
    # args.remove("split_list")
    # kwargs = {"split_list": split_list}
    # for arg in args:
    #     if arg == "tsv_path" and arg not in parameters:
    #         kwargs[arg] = parameters["tsv_directory"]
    #     else:
    #         kwargs[arg] = parameters[arg]

    # if ssda_bool:
    #     kwargs["caps_directory"] = caps_target
    #     kwargs["tsv_path"] = tsv_target_lab
    if "tsv_path" in parameters:
        parameters["tsv_directory"] = parameters["tsv_path"]

    data_config = DataConfig(**parameters)
    split_config = ValidationConfig(**parameters)

    return SplitManager.from_config_class(
        data_config=data_config, split_config=split_config, split_list=split_list
    )


class SplitManager:
    def __init__(
        self,
        caps_directory: Path,
        tsv_directory: Path,
        diagnoses,
        baseline: bool = False,
        valid_longitudinal: bool = False,
        multi_cohort: bool = False,
        split_list: Optional[List[int]] = None,
        n_splits: int = 1,
    ):
        """

        Parameters
        ----------
        caps_director: str (path)
            Path to the caps directory
        tsv_path: str
            Path to the tsv that is going to be split
        diagonoses: List[str]
            List of diagnosis
        baseline: bool
            if True, split only on baseline sessions
        valid_longitudinal: bool
            if True, split validation on longitudinal sessions
        multi-cohort: bool
        split_list: List[str]

        """
        self._check_tsv_path(tsv_directory, multi_cohort)
        self.tsv_path = tsv_directory
        self.caps_dict = self._create_caps_dict(caps_directory, multi_cohort)
        self.multi_cohort = multi_cohort
        self.diagnoses = diagnoses
        self.baseline = baseline
        self.valid_longitudinal = valid_longitudinal
        self.split_list = split_list
        self.n_splits = n_splits

    @classmethod
    def from_config_class(
        cls,
        data_config: DataConfig,
        split_config: ValidationConfig,
        split_list: Optional[List[int]] = None,
    ):
        return cls(
            caps_directory=data_config.caps_directory,
            tsv_directory=split_config.tsv_directory,
            diagnoses=data_config.diagnoses,
            baseline=data_config.baseline,
            valid_longitudinal=split_config.valid_longitudinal,
            multi_cohort=data_config.multi_cohort,
            split_list=split_list,
            n_splits=split_config.n_splits,
        )

    def max_length(self) -> int:
        """Maximum number of splits"""
        return self.n_splits

    def __len__(self):
        if not self.split_list:
            return self.n_splits
        else:
            return len(self.split_list)

    @property
    def allowed_splits_list(self):
        """
        List of possible splits if no restriction was applied

        Returns:
            list[int]: list of all possible splits
        """
        return [i for i in range(self.n_splits)]

    def __getitem__(self, item) -> Dict:
        """
        Returns a dictionary of DataFrames with train and validation data.

        Args:
            item (int): Index of the split wanted.
        Returns:
            Dict[str:pd.DataFrame]: dictionary with two keys (train and validation).
        """
        self._check_item(item)

        if self.multi_cohort:
            tsv_df = pd.read_csv(self.tsv_path, sep="\t")
            train_df = pd.DataFrame()
            valid_df = pd.DataFrame()
            found_diagnoses = set()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.at[idx, "cohort"]
                cohort_path = Path(tsv_df.at[idx, "path"])
                cohort_diagnoses = (
                    tsv_df.at[idx, "diagnoses"].replace(" ", "").split(",")
                )
                if bool(set(cohort_diagnoses) & set(self.diagnoses)):
                    target_diagnoses = list(set(cohort_diagnoses) & set(self.diagnoses))

                    cohort_train_df, cohort_valid_df = self.concatenate_diagnoses(
                        item, cohort_path=cohort_path, cohort_diagnoses=target_diagnoses
                    )
                    cohort_train_df["cohort"] = cohort_name
                    cohort_valid_df["cohort"] = cohort_name
                    train_df = pd.concat([train_df, cohort_train_df])
                    valid_df = pd.concat([valid_df, cohort_valid_df])
                    found_diagnoses = found_diagnoses | (
                        set(cohort_diagnoses) & set(self.diagnoses)
                    )

            if found_diagnoses != set(self.diagnoses):
                raise ValueError(
                    f"The diagnoses found in the multi cohort dataset {found_diagnoses} "
                    f"do not correspond to the diagnoses wanted {set(self.diagnoses)}."
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

    def get_dataframe_from_tsv_path(self, tsv_path: Path) -> pd.DataFrame:
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

    def load_data(
        self, tsv_path: Path, cohort_diagnoses: Optional[List[str]] = None
    ) -> pd.DataFrame:
        df = self.get_dataframe_from_tsv_path(tsv_path)
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
            cohort_diagnoses = self.diagnoses

        tmp_cohort_path = cohort_path if cohort_path is not None else self.tsv_path
        train_path, valid_path = self._get_tsv_paths(
            tmp_cohort_path,
            split,
        )

        logger.debug(f"Training data loaded at {train_path}")
        if self.baseline:
            train_path = train_path / "train_baseline.tsv"
        else:
            train_path = train_path / "train.tsv"
        train_df = self.load_data(train_path, cohort_diagnoses)

        logger.debug(f"Validation data loaded at {valid_path}")
        if self.valid_longitudinal:
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
            return range(self.n_splits)
        else:
            return self.split_list

    def _check_item(self, item):
        if item not in self.allowed_splits_list:
            raise IndexError(
                f"Split index {item} out of allowed splits {self.allowed_splits_list}."
            )

    @staticmethod
    def _create_caps_dict(caps_directory: Path, multi_cohort):
        if multi_cohort:
            if caps_directory.suffix != ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is given, the CAPS_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(caps_directory, sep="\t")
                SplitManager._check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.at[idx, "cohort"]
                    caps_path = caps_df.at[idx, "path"]
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(caps_directory)
            caps_dict = {"single": caps_directory}

        return caps_dict

    @staticmethod
    def _check_tsv_path(tsv_path, multi_cohort):
        if multi_cohort:
            if tsv_path.suffix != ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is given, the TSV_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                tsv_df = pd.read_csv(tsv_path, sep="\t")
                SplitManager._check_multi_cohort_tsv(tsv_df, "labels")
        else:
            if tsv_path.suffix == ".tsv":
                raise ClinicaDLConfigurationError(
                    f"You gave the path to a TSV file in tsv_path {tsv_path}. "
                    f"To use multi-cohort framework, please add 'multi_cohort=true' to the configuration file or the --multi_cohort flag."
                )

    @staticmethod
    def _check_multi_cohort_tsv(tsv_df, purpose):
        if purpose.upper() == "CAPS":
            mandatory_col = {"cohort", "path"}
        else:
            mandatory_col = {"cohort", "path", "diagnoses"}
        if not mandatory_col.issubset(tsv_df.columns.values):
            raise ClinicaDLTSVError(
                f"Columns of the TSV file used for {purpose} location must include {mandatory_col}."
            )
