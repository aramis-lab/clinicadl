from pathlib import Path

from clinicadl.utils.split_manager.split_manager import SplitManager


class SingleSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        split_list=None,
    ):
        super().__init__(
            caps_directory,
            tsv_path,
            diagnoses,
            baseline,
            multi_cohort,
            split_list,
        )

    def max_length(self) -> int:
        return 1

    def __len__(self):
        return 1

    @property
    def allowed_splits_list(self):
        return [0]

    def split_iterator(self):
        return range(1)

    def _get_tsv_paths(self, cohort_path: Path, *args):
        train_path = cohort_path
        valid_path = cohort_path

        return train_path, valid_path
