from typing import Optional

from clinicadl.maps.maps import Maps
from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.split.split import Split
from clinicadl.utils.computational.config import ComputationalConfig

from .base import ClinicaDLConfig


class _TrainingState(ClinicaDLConfig):
    maps: Maps
    metrics: ClinicaDLMetrics
    model: ClinicaDLModel
    optim: OptimizationConfig
    comp: ComputationalConfig
    stop: bool = False
    n_batch: int = -1
    split: Optional[Split] = None
    epoch: int = -1
    batch: int = -1

    def reset(self, split: Split):
        """TO COMPLETE"""
        self.n_batch = len(split.train_loader)
        self.split = split
        self.stop = False
        self.epoch = 0
        self.batch = 0
