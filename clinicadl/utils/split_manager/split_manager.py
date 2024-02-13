import abc
from logging import getLogger
from pathlib import Path

import pandas as pd

from clinicadl.utils.clinica_utils import check_caps_folder
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl.split_manager")


class SplitManager:
    def __init__(
        self,
        caps_directory: Path,
        tsv_path: Path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        split_list=None,
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
        multi-cohort: bool
        split_list: List[str]

        """
        self._check_tsv_path(tsv_path, multi_cohort)
        self.tsv_path = tsv_path
        self.caps_dict = self._create_caps_dict(caps_directory, multi_cohort)
        self.multi_cohort = multi_cohort
        self.diagnoses = diagnoses
        self.baseline = baseline
        self.split_list = split_list

    @abc.abstractmethod
    def max_length(self) -> int:
        """Maximum number of splits"""
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @property
    @abc.abstractmethod
    def allowed_splits_list(self):
        """
        List of possible splits if no restriction was applied

        Returns:
            list[int]: list of all possible splits
        """
        pass

    def __getitem__(self, item):
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
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_path = Path(tsv_df.loc[idx, "path"])
                cohort_diagnoses = (
                    tsv_df.loc[idx, "diagnoses"].replace(" ", "").split(",")
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

    def concatenate_diagnoses(
        self, split, cohort_path: Path = None, cohort_diagnoses=None
    ):
        """Concatenated the diagnoses needed to form the train and validation sets."""
        tmp_cohort_path = cohort_path if cohort_path is not None else self.tsv_path
        train_path, valid_path = self._get_tsv_paths(
            tmp_cohort_path,
            split,
        )
        logger.debug(f"Training data loaded at {train_path}")
        logger.debug(f"Validation data loaded at {valid_path}")
        if cohort_diagnoses is None:
            cohort_diagnoses = self.diagnoses
        if self.baseline:
            train_path = train_path / "train_baseline.tsv"
        else:
            train_path = train_path / "train.tsv"

        valid_path = valid_path / "validation_baseline.tsv"

        train_df = pd.read_csv(train_path, sep="\t")
        valid_df = pd.read_csv(valid_path, sep="\t")

        list_columns = train_df.columns.values

        if (
            "diagnosis"
            not in list_columns
            # or "age" not in list_columns
            # or "sex" not in list_columns
        ):
            parents_path = train_path.resolve().parent
            while (
                not (parents_path / "labels.tsv").is_file()
                and ((parents_path / "kfold.json").is_file())
                or (parents_path / "split.json").is_file()
            ):
                parents_path = parents_path.parent
            try:
                labels_df = pd.read_csv(parents_path / "labels.tsv", sep="\t")
                train_df = pd.merge(
                    train_df,
                    labels_df,
                    how="inner",
                    on=["participant_id", "session_id"],
                )
            except:
                pass

        list_columns = valid_df.columns.values
        if (
            "diagnosis"
            not in list_columns
            # or "age" not in list_columns
            # or "sex" not in list_columns
        ):
            parents_path = valid_path.resolve().parent
            while (
                not (parents_path / "labels.tsv").is_file()
                and ((parents_path / "kfold.json").is_file())
                or (parents_path / "split.json").is_file()
            ):
                parents_path = parents_path.parent
            try:
                labels_df = pd.read_csv(parents_path / "labels.tsv", sep="\t")
                valid_df = pd.merge(
                    valid_df,
                    labels_df,
                    how="inner",
                    on=["participant_id", "session_id"],
                )
            except:
                pass
        train_df = train_df[
            train_df.diagnosis.isin(cohort_diagnoses)
        ]  # TO MODIFY with train
        valid_df = valid_df[valid_df.diagnosis.isin(cohort_diagnoses)]

        train_df.reset_index(inplace=True, drop=True)
        valid_df.reset_index(inplace=True, drop=True)

        return train_df, valid_df

    @abc.abstractmethod
    def _get_tsv_paths(self, cohort_path, *args):
        """
        Computes the paths to the TSV files needed depending on the split structure.

        Args:
            cohort_path (str): path to the split structure of a cohort.
            split (int): Index of the split.
        Returns:
            train_path (str): path to the directory containing training data.
            valid_path (str): path to the directory containing validation data.
        """
        pass

    @abc.abstractmethod
    def split_iterator(self):
        """Returns an iterable to iterate on all splits wanted."""
        pass

    def _check_item(self, item):
        if item not in self.allowed_splits_list:
            raise IndexError(
                f"Split index {item} out of allowed splits {self.allowed_splits_list}."
            )

    @staticmethod
    def _create_caps_dict(caps_directory: Path, multi_cohort):
        if multi_cohort:
            if not caps_directory.suffix == ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is given, the CAPS_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(caps_directory, sep="\t")
                SplitManager._check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.loc[idx, "cohort"]
                    caps_path = caps_df.loc[idx, "path"]
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(caps_directory)
            caps_dict = {"single": caps_directory}

        return caps_dict

    @staticmethod
    def _check_tsv_path(tsv_path, multi_cohort):
        if multi_cohort:
            if not tsv_path.suffix == ".tsv":
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
