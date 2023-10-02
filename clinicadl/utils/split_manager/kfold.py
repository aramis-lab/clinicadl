from pathlib import Path

from clinicadl.utils.split_manager.split_manager import SplitManager


class KFoldSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        n_splits,
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
        self.n_splits = n_splits

    def max_length(self) -> int:
        return self.n_splits

    def __len__(self):
        if not self.split_list:
            return self.n_splits
        else:
            return len(self.split_list)

    @property
    def allowed_splits_list(self):
        return [i for i in range(self.n_splits)]

    def split_iterator(self):
        if not self.split_list:
            return range(self.n_splits)
        else:
            return self.split_list

    def _get_tsv_paths(self, cohort_path: Path, *args):
        for split in args:
            train_path = cohort_path / f"split-{split}"
            valid_path = cohort_path / f"split-{split}"
        return train_path, valid_path
