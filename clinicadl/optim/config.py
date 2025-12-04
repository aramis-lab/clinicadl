from __future__ import annotations

from pydantic import PositiveInt

from clinicadl.utils.config import ClinicaDLConfig


class OptimizationConfig(ClinicaDLConfig):
    """
    Config class to configure the optimization process.

    Parameters
    ----------
    epochs : PositiveInt, default=10
        Number of epochs.
    accumulation_steps : PositiveInt, default=1
        The number of loss computations during which gradients are accumulated before performing the weights update.
        This allows to virtually increase the size of the batch.
    evaluation_steps : PositiveInt, default=1
        Perform evaluation on the validation every x epochs.
    """

    epochs: PositiveInt = 10
    accumulation_steps: PositiveInt = 1
    evaluation_steps: PositiveInt = 1
