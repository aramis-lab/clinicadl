from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional

import pandas as pd

from clinicadl.utils.dictionary.words import BATCH, EPOCH

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.train import TrainerState


class TrainingLossCallback(Callback):
    """
    To record batch training losses in a :py:class:`pd.DataFrame`,
    and save them in a ``TSV`` file at the end of training.
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None

    # pylint: disable=arguments-differ, unused-argument
    def on_train_start(self, *, model: Model, **kwargs) -> None:
        losses = list(model.get_loss_functions().keys())
        self.df = pd.DataFrame(columns=[EPOCH, BATCH] + losses)
        self.df = self.df.set_index([EPOCH, BATCH]).astype(dtype=float)

    def on_backward_step_start(
        self,
        *,
        state: TrainerState,
        loss: LossType,
        **kwargs,
    ) -> None:
        if isinstance(loss, dict):
            losses = {col: value.item() for col, value in loss.items()}

        else:
            losses = {self.df.columns[0]: loss.item()}

        for name, value in losses.items():
            self.df.at[(state.current_epoch, state.current_train_batch), name] = value

    def on_train_end(self, *, maps: Maps, state: TrainerState, **kwargs) -> None:
        maps.training.splits[state.split_idx].logs.create(exist_ok=True)
        maps.save_file(
            self.df,
            path=maps.training.splits[state.split_idx].logs.training_loss_tsv,
            overwrite=True,
        )

    def state_dict(self) -> Mapping[str, Any]:
        if self.df is None:
            return {}
        return self.df.to_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.df = pd.DataFrame.from_dict(state_dict)
