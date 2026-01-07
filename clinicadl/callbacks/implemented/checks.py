from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model


class ChecksCallback(Callback):
    """
    Performs various checks on the objects defined by the user.
    """

    def __init__(self):
        self._check_losses = _CheckLosses()

    def on_train_start(self, **kwargs) -> None:
        self._check_losses.on_train_start(**kwargs)

    def on_backward_step_start(self, **kwargs) -> None:
        self._check_losses.on_backward_step_start(**kwargs)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


class _CheckLosses:
    """
    Checks the loss functions handled by the :py:class:`clinicadl.models.Model`.
    """

    def __init__(self):
        self._checked = False
        self._losses: list[str] = []

    def on_train_start(self, *, model: Model, **kwargs) -> None:
        """
        Checks that :py:meth:`clinicadl.models.Model.get_loss_functions` returns a dictionary with at least one key.
        """
        self._checked = False

        losses = model.get_loss_functions()
        if not (isinstance(losses, dict) and losses):
            raise ValueError(
                f"clinicadl.models.Model.get_loss_functions method should return a dictionary with at least one key. Got: {losses}"
            )

        self._losses = sorted(list(losses.keys()))

    def on_backward_step_start(
        self,
        *,
        loss: LossType,
        **kwargs,
    ) -> None:
        """
        Checks consistency between :py:meth:`clinicadl.models.Model.get_loss_functions` and
        :py:meth:`clinicadl.models.Model.forward_step`.
        """
        if self._checked:
            return

        if isinstance(loss, torch.Tensor):
            if len(self._losses) > 1:
                raise ValueError(
                    f"clinicadl.models.Model.forward_step returns a single loss, whereas clinicadl.models.Model.get_loss_functions "
                    f"returns {len(self._losses)} loss function(s) {self._losses}"
                )
            self._checked = True
            return

        elif isinstance(loss, dict):
            if all(isinstance(value, torch.Tensor) for value in loss.values()):
                if self._losses != (keys := sorted(list(loss.keys()))):
                    raise ValueError(
                        f"clinicadl.models.Model.forward_step returns loss(es) named {keys}, whereas clinicadl.models.Model.get_loss_functions "
                        f"returns {self._losses} loss function(s)"
                    )
                self._checked = True
                return

        raise ValueError(
            f"clinicadl.models.Model.forward_step should return a Tensor, or a dict of Tensors. Got: {loss}"
        )
