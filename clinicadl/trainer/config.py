from clinicadl.maps.maps import Maps
from clinicadl.metrics.metrics import ClinicaDLMetrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.config import ClinicaDLConfig


class _TrainingConfig(ClinicaDLConfig):
    maps: Maps
    metrics: ClinicaDLMetrics
    model: ClinicaDLModel
    optim: OptimizationConfig
    comp: ComputationalConfig
    split: int = -1
    epoch: int = -1
    batch: int = -1

    def reset(self, split: int):
        self.split = split
        self.epoch = 0
        self.batch = 0
