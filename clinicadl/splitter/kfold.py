from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import NonNegativeInt

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.utils import tsv_to_df
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.splitter.split_output import Split
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal
from clinicadl.utils.exceptions import ClinicaDLTSVError

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class KFolder:
    def __init__(self, caps_dataset: CapsDataset, manager: ExperimentManager) -> None:
        self.dataset = caps_dataset
        self.reader = caps_dataset.caps_reader
        self.df = caps_dataset.df
        self.manager = manager

    def make_splits(
        self,
        n_splits: NonNegativeInt = 5,
        subset_name: str = "validation",
        stratification: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        additional_data_tsv: Optional[Path] = None,
        ignore_demographis: bool = False,
    ) -> Path:
        """
        Performs a k-fold split for each label independently on the subject level.
        The output (the tsv file) will have two new columns :
            - split, with the number of the split the subject is in.
            - datagroup, with the name of the group (train or subset_name) the subject is in.

        The train group will contain baseline and longitudinal sessions,
        whereas the test group will only include the baseline sessions for each split.

        Parameters
        ----------
        n_splits: int > 1
            Number of splits in the k-fold cross-validation.
        subset_name: str
            Name of the subset that is complementary to train.
        stratification: str
            Name of variables used to stratify k-fold.
        merged_tsv: str
            Path to the merged.tsv file, output of clinica iotools merge-tsv.
        """

        if n_splits <= 1:
            raise ValueError("Number of splits must be greater than 1")

        output_dir = self._get_results_directory(n_splits, output_dir)
        df = self._merge_df(additional_data_tsv=additional_data_tsv)
        stratification = self._get_stratification(
            df, ignore_demographics=ignore_demographis, stratification=stratification
        )

        self._create_splits(n_splits, subset_name, stratification)

        return output_dir

    def _merge_df(self, additional_data_tsv: Optional[Path] = None) -> pd.DataFrame:
        if additional_data_tsv:
            additional_data_df = tsv_to_df(additional_data_tsv)
            self.df.set_index([PARTICIPANT_ID, SESSION_ID])
            keys_df1 = set(self.df.index)

            additional_data_df.set_index([PARTICIPANT_ID, SESSION_ID])
            keys_df2 = set(additional_data_df.index)

            if keys_df1 - keys_df2 != 0:
                raise ValueError(
                    "Additional data does not contain the same participants and sessions as the original data"
                )

            df = pd.merge(
                self.df, additional_data_df, how="left", on=[PARTICIPANT_ID, SESSION_ID]
            )

        return df

    def _create_splits(
        self,
        split_label: str,
        n_splits: NonNegativeInt,
        subset_name: str,
        results_directory: Path,
    ):
        """
        Split data at the subject-level in training and test to have equivalent distributions in split_label.
        Writes test and train Dataframes.

        Parameters
        ----------
        diagnosis_df: Dataframe
            Columns must include ['participant_id', 'session_id', 'diagnosis']
        split_label: str
            Label on which the split is done (categorical variables)
        n_splits: int
            Number of splits in the k-fold cross-validation.
        subset_name: str
            Name of the subset split.
        results_directory: str (path)
            Path to the results directory.

        """

        baseline_df = extract_baseline(self.df)

        if stratification is None:
            diagnoses_list = list(baseline_df["diagnosis"])
            unique = list(set(diagnoses_list))
            y = np.array([unique.index(x) for x in diagnoses_list])
        else:
            stratification_list = list(baseline_df[split_label])
            unique = list(set(stratification_list))
            y = np.array([unique.index(x) for x in stratification_list])

        splits = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=2)

        for i, indices in enumerate(splits.split(np.zeros(len(y)), y)):
            train_index, test_index = indices

            train_df = baseline_df.iloc[train_index]
            long_train_df = retrieve_longitudinal(train_df, diagnosis_df)
            train_df.reset_index(inplace=True, drop=True)

            test_df = baseline_df.iloc[test_index]

            # train_df = train_df[["participant_id", "session_id"]]
            # test_df = test_df[["participant_id", "session_id"]]
            # long_train_df = long_train_df[["participant_id", "session_id"]]

            (results_directory / f"split-{i}").mkdir(parents=True)

            train_df.to_csv(
                results_directory / f"split-{i}" / "train_baseline.tsv",
                sep="\t",
                index=False,
            )
            test_df.to_csv(
                results_directory / f"split-{i}" / f"{subset_name}_baseline.tsv",
                sep="\t",
                index=False,
            )

            long_train_df.to_csv(
                results_directory / f"split-{i}" / "train.tsv",
                sep="\t",
                index=False,
            )
            if valid_longitudinal:
                long_test_df = retrieve_longitudinal(test_df, diagnosis_df)
                test_df.reset_index(inplace=True, drop=True)

                long_test_df.to_csv(
                    results_directory / f"split-{i}" / f"{subset_name}.tsv",
                    sep="\t",
                    index=False,
                )
