from logging import getLogger

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import sampler
from torch.utils.data.distributed import DistributedSampler

from clinicadl.caps_dataset.data import CapsDataset
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.task_manager.task_manager import TaskManager

logger = getLogger("clinicadl.task_manager")


class ClassificationManager(TaskManager):
    def __init__(
        self,
        mode,
        n_classes=None,
        df=None,
        label=None,
    ):
        if n_classes is None:
            n_classes = self.output_size(None, df, label)
        self.n_classes = n_classes
        super().__init__(mode, n_classes)
