from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

import pandas as pd
import torch

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import BATCH, EPOCH
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.train import TrainerState


class TrainingLossCallbackConfig(ObjectConfig["TrainingLossCallback"]):
    """Config class for ``TrainingLossCallback``."""

    @classmethod
    def _get_class(cls):
        return TrainingLossCallback


class TrainingLossCallback(Callback, HasConfig[TrainingLossCallbackConfig]):
    """
    To record batch training losses in a :py:class:`pd.DataFrame`,
    and save them in a ``TSV`` file at the end of training.
    """

    _config_type = TrainingLossCallbackConfig

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.config = self._config_type()

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(self, model: Model, **kwargs) -> None:
        losses = list(model.get_loss_functions().keys())
        assert losses, "get_loss_functions method of you clinicadl.models.Model should return a dictionary with at least one key."
        self.df = pd.DataFrame(columns=[EPOCH, BATCH] + losses)
        self.df = self.df.set_index([EPOCH, BATCH]).astype(dtype=float)

    def on_backward_step_start(
        self,
        *,
        state: TrainerState,
        loss: LossType,
    ) -> None:
        if isinstance(loss, torch.Tensor):
            if len(self.df.columns) > 1:
                raise ValueError(
                    f"clinicadl.models.Model.forward_step returns a single loss, whereas clinicadl.models.Model.get_loss_functions "
                    f"returns {len(self.df.columns)} loss function(s) {sorted(self.df.columns.to_list())}"
                )
            losses = {self.df.columns[0]: loss.item()}

        elif isinstance(loss, dict):
            assert all(
                isinstance(value, torch.Tensor) for value in loss.values()
            ), f"forward_step should return a Tensor, or a dict of Tensors. Got: {loss}"
            losses = {col: value.item() for col, value in loss.items()}

        else:
            raise ValueError(
                f"forward_step should return a Tensor, or a dict of Tensors. Got: {loss}"
            )

        if (keys := set(losses.keys())) != (
            excepted_keys := set(self.df.columns.to_list())
        ):
            raise ValueError(
                f"clinicadl.models.Model.forward_step returns loss(es) named {sorted(list(keys))}, whereas clinicadl.models.Model.get_loss_functions "
                f"returns {sorted(list(excepted_keys))} loss function(s)"
            )

        for name, value in losses.items():
            self.df.at[(state.current_epoch, state.current_train_batch), name] = value

    def on_train_end(self, maps: Maps, state: TrainerState, **kwargs) -> None:
        maps.save_file(
            self.df,
            path=maps.training.splits[state.split_idx].logs.training_loss,
            overwrite=True,
        )

    def state_dict(self) -> Mapping[str, Any]:
        if self.df is None:
            return {}
        return self.df.to_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.df = pd.DataFrame.from_dict(state_dict)
