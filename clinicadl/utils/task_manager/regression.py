import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler

from clinicadl.caps_dataset.data import CapsDataset
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.task_manager.task_manager import TaskManager


class RegressionManager(TaskManager):
    def __init__(
        self,
        mode,
    ):
        super().__init__(mode)
