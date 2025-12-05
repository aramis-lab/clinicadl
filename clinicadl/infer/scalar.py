from typing import Any

import torch
import torch.nn as nn

from clinicadl.utils.enum import BaseEnum

from .base import Inferer


class ScalarInferenceMode(BaseEnum):
    AVERAGE = "average"  # mean of outputs
    MEDIAN = "median"  # median of outputs
    VOTE = "vote"  # majority vote


class ScalarInferer(Inferer):
    # def __init__(self, mode: ScalarInferenceMode):
    def __call__(
        self, x: torch.Tensor, network: nn.Module, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        pass
