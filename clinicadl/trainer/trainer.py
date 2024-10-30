from pathlib import Path

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.splitter.kfold import Split


class Trainer:
    def __init__(self) -> None:
        """TO COMPLETE"""

    @classmethod
    def from_json(cls, config_file: Path, manager: ExperimentManager) -> Trainer:
        """TO COMPLETE"""
        return cls()

    def train(self, model: ClinicaDLModel, split: Split):
        """TO COMPLETE"""
        pass
