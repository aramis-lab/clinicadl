from os import path

from clinicadl.utils.split_manager.split_manager import SplitManager


class SingleSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        folds=None,
    ):
        super().__init__(
            caps_directory, tsv_path, diagnoses, baseline, multi_cohort, folds
        )

    def max_length(self) -> int:
        return 1

    def __len__(self):
        return 1

    @property
    def allowed_folds_list(self):
        return [0]

    def fold_iterator(self):
        return range(1)

    def _get_tsv_paths(self, cohort_path, fold):
        train_path = path.join(cohort_path, "train")
        valid_path = path.join(cohort_path, "validation")
        return train_path, valid_path
