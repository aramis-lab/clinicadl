from typing import Optional

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager


class Split:
    def __init__(
        self,
    ):
        """TO COMPLETE"""
        pass


class KFolder:
    def __init__(
        self, n_splits: int, caps_dataset: CapsDataset, manager: ExperimentManager
    ) -> None:
        """TO COMPLETE"""

    def split_iterator(self, split_list: Optional[list] = None) -> list[Split]:
        """TO COMPLETE"""

        return list[Split()]
