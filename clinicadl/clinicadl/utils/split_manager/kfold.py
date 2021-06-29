from os import path

from clinicadl.utils.split_manager.template import SplitManager


class KFoldSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        n_splits,
        baseline=False,
        multi_cohort=False,
        folds=None,
        logger=None,
    ):
        super().__init__(
            caps_directory, tsv_path, diagnoses, baseline, multi_cohort, folds, logger
        )
        self.n_splits = n_splits

    def __len__(self):
        if self.folds is None:
            return self.n_splits
        else:
            return len(self.folds)

    def fold_iterator(self):
        if self.folds is None:
            return range(self.n_splits)
        else:
            return self.folds

    def _get_tsv_paths(self, item):
        train_path = path.join(
            self.tsv_path, f"train_splits-{self.n_splits}", f"split-{item}"
        )
        valid_path = path.join(
            self.tsv_path, f"validation_splits-{self.n_splits}", f"split-{item}"
        )
        return train_path, valid_path

    def _check_folds(self):
        possible_folds = {i for i in range(self.n_splits)}
        if self.folds is not None or set(self.folds).issubset(possible_folds):
            raise ValueError(
                f"Folds list is set to {self.folds}. "
                f"Please use only folds in {possible_folds}."
            )
