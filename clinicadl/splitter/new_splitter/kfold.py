import json
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import NonNegativeInt, PositiveInt
from sklearn.model_selection import StratifiedKFold

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.splitter.new_splitter.split import Split
from clinicadl.splitter.new_splitter.utils import KFoldConfig, SubjectsSessionsSplit
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal
from clinicadl.utils.exceptions import ClinicaDLTSVError


class KFold:
    """
    Handles K-Fold cross-validation with optional stratification and demographic balancing.
    Allows saving, reading, and iterating over splits for reproducibility.
    """

    json_name = "kfold.json"

    def __init__(self, dataset: CapsDataset):
        """
        Initialize KFold with a dataset.

        Parameters
        ----------
        dataset : CapsDataset
            Dataset to split for cross-validation.
        """
        self.dataset = dataset
        self.df = dataset.df
        self.subjects_sessions_split: List[SubjectsSessionsSplit] = []
        self.config: Optional[KFoldConfig] = None

    @staticmethod
    def _check_stratification(
        df: pd.DataFrame,
        ignore_demographics: bool,
        stratification: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Checks and validates the specified stratification columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        ignore_demographics : bool
            If True, ignore demographic columns for balancing.
        stratification : List[str], optional
            List of columns to stratify on.

        Returns
        -------
        List[str], optional
            Validated list of stratification columns or None if no stratification is applied.

        Raises
        ------
        ValueError
            If specified stratification columns are missing or if stratification conflicts with demographic handling.
        ClinicaDLTSVError
            If required demographic columns ('age', 'sex') are missing when not ignored.
        """

        if stratification:
            missing_columns = set(stratification) - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Stratification variables {missing_columns} not found in dataset."
                )
            if ignore_demographics:
                raise ValueError("Cannot stratify while ignoring demographics.")

            if not {"age", "sex"}.issubset(df.columns):
                raise ClinicaDLTSVError(
                    "Dataset missing 'age' or 'sex' columns for demographic balancing."
                )
            # TODO: check if we want to always stratify on age and sex

        elif not ignore_demographics:
            stratification = ["age", "sex"]
        return stratification

    @staticmethod
    def preprocess_stratification(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        ignore_demographics: bool = False,
    ) -> List[str]:
        """
        Preprocess stratification columns by creating labels for each subject.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        columns : Optional[List[str]]
            Columns to stratify on.
        ignore_demographics : bool
            If True, ignore demographic columns.

        Returns
        -------
        List[str]
            List of stratification labels for the dataset.
        """
        columns = KFold._check_stratification(df, ignore_demographics, columns)
        if not columns:
            return ["0"] * len(df)

        labels = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numerical column: bin into 5 equal groups or fewer if unique values < 5
                labels.append(
                    pd.cut(
                        df[col], bins=min(5, df[col].nunique()), labels=False
                    ).astype(str)
                )
            else:
                labels.append(df[col].astype(str))
        return ["_".join(label) for label in zip(*labels)]

    def make_splits(
        self,
        n_splits: PositiveInt = 5,
        stratification: Optional[List[str]] = None,
        ignore_demographics: bool = False,
    ) -> None:
        """
        Perform K-Fold splitting with optional stratification.

        Parameters
        ----------
        n_splits : PositiveInt, default=5
            Number of splits.
        stratification : Optional[List[str]]
            Columns to use for stratification.
        ignore_demographics : bool, default=False
            If True, demographic balancing is ignored.

        Returns
        -------
        None
            Populates the `subjects_sessions_split` attribute with the generated splits.
        """

        self.config = KFoldConfig(
            n_splits=n_splits,
            stratification=stratification,
            ignore_demographics=ignore_demographics,
        )

        baseline_df = extract_baseline(self.df)
        stratify_labels = self.preprocess_stratification(
            df=baseline_df,
            columns=stratification,
            ignore_demographics=ignore_demographics,
        )

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)

        self.subjects_sessions_split = [
            SubjectsSessionsSplit(
                train=baseline_df.iloc[train_idx],
                validation=baseline_df.iloc[test_idx],
            )
            for train_idx, test_idx in skf.split(baseline_df, stratify_labels)
        ]

    def write(
        self,
        output_dir: Path,
        subset_name: str = "validation",
        valid_longitudinal: bool = False,
    ) -> None:
        """
        Save splits and configuration to files.
        Should be called after `make_splits`.

        Parameters
        ----------
        output_dir : Path
            Directory to save split files.
        subset_name : str, default="validation"
            Name of the validation subset.
        valid_longitudinal : bool, default=False
            If True, includes longitudinal data for validation.

        """

        if not self.subjects_sessions_split or self.config is None:
            raise ValueError("No splits available. Run 'make_splits' first.")

        for i, split in enumerate(self.subjects_sessions_split):
            split_dir = output_dir / f"split-{i}"
            split_dir.mkdir(parents=True, exist_ok=True)

            self._write_to_csv(split.train, split_dir / "train_baseline.tsv")
            self._write_to_csv(
                split.validation, split_dir / f"{subset_name}_baseline.tsv"
            )

            long_train_df = retrieve_longitudinal(split.train, self.df)
            self._write_to_csv(long_train_df, split_dir / "train.tsv")

            if valid_longitudinal:
                long_val_df = retrieve_longitudinal(split.validation, self.df)
                self._write_to_csv(long_val_df, split_dir / f"{subset_name}.tsv")

        self.config = self.config.model_copy(
            update={
                "subset_name": subset_name,
                "valid_longitudinal": valid_longitudinal,
                "split_dir": output_dir,
            }
        )

        self._write_json()

    def _write_to_csv(self, df: pd.DataFrame, file_path: Path) -> None:
        """
        Save DataFrame to a TSV file.

        Parameters
        ----------
        df : pd.DataFrame
            Data to save.
        file_path : Path
            Destination file path.

        """
        if file_path.is_file():
            raise FileExistsError(f"File {file_path} already exists.")
        df.reset_index(drop=True, inplace=True)
        df.to_csv(file_path, sep="\t", index=False)

    def _write_json(self) -> None:
        """
        Save KFold configuration to JSON.
        """
        if not self.config or not self.config.split_dir:
            raise ValueError(
                "No split directory specified, use the method 'write' to save your splits."
            )

        out_json_file = self.config.split_dir / self.json_name
        if out_json_file.is_file():
            raise FileExistsError(
                f"File {out_json_file} already exists, your splits may have already been written."
            )

        with out_json_file.open(mode="w") as json_file:
            json.dump(self.config.model_dump(), json_file)

    def _read_json(self, split_dir: Path):
        """
        Load KFold configuration from a JSON file.

        Parameters
        ----------
        split_dir : Path
            Directory containing the JSON configuration file.

        Returns
        -------
        KFoldConfig
            The configuration object loaded from the JSON file.
        """
        if not split_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {split_dir}")

        json_file = split_dir / self.json_name

        if not json_file.is_file():
            raise FileNotFoundError(f"No such file: {json_file}")

        with json_file.open(mode="r") as file:
            dict_ = json.load(file)

            return KFoldConfig(**dict_)

    def _read_split(self, split_dir: Path, split_number: int) -> SubjectsSessionsSplit:
        """
        Load a single split's train and validation sets from files.

        Parameters
        ----------
        split_dir : Path
            Directory containing split data.
        split_number : int
            The split index to load.

        Returns
        -------
        SubjectsSessionsSplit
            Object containing train and validation sets as DataFrames.
        """
        split_path = split_dir / f"split-{split_number}"
        if not split_path.is_dir():
            raise FileNotFoundError(f"No such directory: {split_path}")

        try:
            train = pd.read_csv(split_path / "train_baseline.tsv", sep="\t")
            validation = pd.read_csv(
                split_path / f"{self.config.subset_name}_baseline.tsv", sep="\t"
            )  # type: ignore

        except FileNotFoundError:
            raise FileNotFoundError(
                f"One or more of the required files are missing: 'train_baseline.tsv', '{self.config.subset_name}_baseline.tsv'"
            )  # type: ignore

        return SubjectsSessionsSplit(
            train=train,
            validation=validation,
        )

    def read(self, split_dir: Path) -> None:
        """
        Load all splits and configuration from a directory.

        Parameters
        ----------
        split_dir : Path
            Directory containing the splits and configuration JSON file.

        Returns
        -------
        None
            Populates `subjects_sessions_split` and `config` attributes.
        """

        config_path = split_dir / self.json_name
        with config_path.open("r") as f:
            self.config = KFoldConfig(**json.load(f))

        self.subjects_sessions_split = [
            self._read_split(split_dir, i) for i in range(self.config.n_splits)
        ]

    def get_splits(self, splits: Sequence[int]) -> Generator[Split, None, None]:
        """
        Yield dataset splits by their indices.

        Parameters
        ----------
        splits : Sequence[int]
            Indices of the splits to retrieve.

        Yields
        ------
        Split
            The train and validation datasets for each requested split.

        Raises
        ------
        ValueError
            If the requested split indices are out of range or no splits are available.
        """

        if not self.config or self.config.n_splits is None:
            raise ValueError(
                "No splits found, you must first run the method 'make_splits' to split your dataset, "
                "or use the method 'read' to recover old splits."
            )
        for split in splits:
            if split not in range(self.config.n_splits):
                raise ValueError(
                    f"Split-{split} doesn't exist. There are {self.config.n_splits} splits, numbered from 0 to {self.config.n_splits-1}."
                )
            yield self._get_split(split)

    def _get_split(
        self,
        split_id: int,
    ) -> Split:
        """
        Retrieve a single dataset split.

        Parameters
        ----------
        split_id : int
            Index of the split to retrieve.

        Returns
        -------
        Split
            Object containing train and validation datasets for the specified split.
        """
        subjects_sessions = self.subjects_sessions_split[split_id]
        return Split(
            index=split_id,
            train_dataset=self.dataset.subset(subjects_sessions.train),
            val_dataset=self.dataset.subset(subjects_sessions.validation),
        )
