from pathlib import Path

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.splitter.kfold import Split


def split_tsv(sub_ses_tsv: Path) -> Path:
    """TO COMPLETE"""

    split_dir = Path("")
    return split_dir


def get_single_split(
    n_subject_validation: int, caps_dataset: CapsDataset, manager: ExperimentManager
) -> Split:
    pass
