from os import path

from clinicadl.utils.split_manager.template import SplitManager


class SingleSplit(SplitManager):
    def __init__(
        self,
        caps_directory,
        tsv_path,
        diagnoses,
        baseline=False,
        multi_cohort=False,
        folds=None,
        logger=None,
    ):
        super().__init__(
            caps_directory, tsv_path, diagnoses, baseline, multi_cohort, folds, logger
        )

    def __len__(self):
        return 1

    def fold_iterator(self):
        return range(1)

    def _get_tsv_paths(self, item):
        train_path = path.join(self.tsv_path, "train")
        valid_path = path.join(self.tsv_path, "validation")
        return train_path, valid_path

    def _check_folds(self):
        if self.folds is not None or set(self.folds) != set([0]):
            raise ValueError(
                "Single Split will only perform one fold. "
                "Please do not specify any folds or [0]."
            )
