from __future__ import annotations

from pydantic import PositiveInt

from clinicadl.utils.config import ClinicaDLConfig


class OptimizationConfig(ClinicaDLConfig):
    """
    Config class to configure the optimization process.

    Parameters
    ----------
    num_epochs : PositiveInt, default=10
        Number of epochs.
    accumulation_steps : PositiveInt, default=1
        The number of loss computations for which gradients are accumulated before performing the weights update.
        This allows to virtually increase the size of the batch.
    evaluation_interval : PositiveInt, default=1
        Perform evaluation on the validation every x epochs. For example, if ``evaluation_interval=2``, evaluation
        will be performed after epochs 2, 4, 6, 8, etc.
    """

    num_epochs: PositiveInt = 10
    accumulation_steps: PositiveInt = 1
    evaluation_interval: PositiveInt = 1
