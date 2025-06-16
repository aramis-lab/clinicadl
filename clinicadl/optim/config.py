from __future__ import annotations

from pydantic import PositiveInt

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.typing import PathType


class OptimizationConfig(ClinicaDLConfig):
    """Config class to configure the optimization process."""

    accumulation_steps: PositiveInt = 1  # gives the number of iterations during which gradients are accumulated before performing the weights update. This allows to virtually increase the size of the batch. Default: 1.
    evaluation_steps: PositiveInt = 5  # gives the number of iterations to perform an evaluation internal to an epoch. Default will only perform an evaluation at the end of each epoch.
    epochs: PositiveInt = 10
