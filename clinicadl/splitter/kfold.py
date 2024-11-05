from typing import Optional

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager


class Split:
    def __init__(
        self,
        caps_dataset: CapsDataset,
        manager: ExperimentManager,
        n_splits: int = 1,
    ) -> None:
        """TO COMPLETE"""

        self.dataset = caps_dataset
        self.manager = manager

    def split_iterator(self, start, end):
        get_single_split(
            n_subject_validation=0,
            caps_dataset=dataset_multi_modality_multi_extract,
            manager=manager,
        )


class KFolder:
    def __init__(
        self, n_splits: int, caps_dataset: CapsDataset, manager: ExperimentManager
    ) -> None:
        """TO COMPLETE"""
        self.dataset = caps_dataset
        self.manager = manager
        self.n_splits = n_splits

    def split_iterator(self, split_list: Optional[list] = None) -> list[Split]:
        """TO COMPLETE"""

        return list[Split()]

    def __getitem__(self, key):
        pass


class Splitter:
    def __init__(
        self, n_splits: int, caps_dataset: CapsDataset, manager: ExperimentManager
    ) -> None:
        """TO COMPLETE"""
        self.dataset = caps_dataset
        self.manager = manager
        self.n_splits = n_splits

    def split_iterator(self, split_list: Optional[list] = None) -> list[Split]:
        """TO COMPLETE"""

        return list[Split()]

    def __getitem__(self, key):
        pass
