from typing import Optional

from clinicadl.maps.maps import Maps
from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.splitter.split import Split
from clinicadl.utils.computational.config import ComputationalConfig

from .base import ClinicaDLConfig


class _TrainingState(ClinicaDLConfig):
    maps: Maps
    metrics: ClinicaDLMetrics
    model: ClinicaDLModel
    optim: OptimizationConfig
    comp: ComputationalConfig
    split: Optional[Split] = None
    epoch: int = -1
    batch: int = -1

    def reset(self, split: Split):
        """TO COMPLETE"""
        self.split = split
        self.epoch = 0
        self.batch = 0
