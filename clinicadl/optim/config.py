from __future__ import annotations

from typing import Optional

from pydantic import NonNegativeFloat, PositiveInt

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
        To perform evaluation on the validation every x epochs. For example, if ``evaluation_interval=2``, evaluation
        will be performed after epochs #2, #4, #6, #8, etc.
    clip_grad_norm : Optional[NonNegativeFloat], default=None
        To clip gradient norm. Give here the maximum norm accepted (a non-negative float).
        The norm is computed over the norms of the individual gradients of all parameters (see
        :py:func:`torch.nn.utils.clip_grad_norm_`).
    grad_norm_type : float, default=2
        Indice p of the p-norm for gradient norm clipping. Can be any float.
    clip_grad_value : Optional[NonNegativeFloat]
        For individual gradients clipping (see :py:func:`torch.nn.utils.clip_grad_value_`). Give here the maximum absolute value accepted (a non-negative float).
        Contrary to norm clipping, value clipping is performed on each gradient individually.

        .. important::
            Gradient value clipping is performed before gradient norm clipping.
    reset_model : ResetOption, default="all"
        Whether to reset the neural network weights before a training phase. If ``True``,
        all the trainable parameters (i.e. with ``requires_grad=True``) will be randomly reset.
        Otherwise, the weights are not reset, so the weights initially passed when instantiating
        the :py:class:`~clinicadl.train.Trainer` is always used as a starting point
    """

    num_epochs: PositiveInt = 10
    accumulation_steps: PositiveInt = 1
    evaluation_interval: PositiveInt = 1
    clip_grad_norm: Optional[NonNegativeFloat] = None
    grad_norm_type: float = 2
    clip_grad_value: Optional[NonNegativeFloat] = None
    reset_model: bool = True
