from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd
import torch

from clinicadl.utils.dictionary.words import PARTICIPANT_ID
from clinicadl.utils.exceptions import DataFrameError, DataLeakageError
from clinicadl.utils.tsvtools import read_data

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import DataLoader
    from clinicadl.io import Maps
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.split import Split


class ChecksCallback(Callback):
    """
    Performs various checks on the objects defined by the user.
    """

    def __init__(self):
        self._check_dataframes = _CheckDataFrames()
        self._check_data_leakage = _CheckDataLeakage()
        self._check_losses = _CheckLosses()

    def on_train_start(self, **kwargs) -> None:
        self._check_dataframes.on_train_start(**kwargs)
        self._check_data_leakage.on_train_start(**kwargs)
        self._check_losses.on_train_start(**kwargs)

    def on_backward_step_start(self, **kwargs) -> None:
        self._check_losses.on_backward_step_start(**kwargs)

    def on_test_start(self, **kwargs) -> None:
        self._check_dataframes.on_test_start(**kwargs)
        self._check_data_leakage.on_test_start(**kwargs)

    def on_predict_start(self, **kwargs) -> None:
        self._check_dataframes.on_predict_start(**kwargs)

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


class _CheckDataFrames:
    """
    Checks that that dataset DataFrames are valid.
    """

    def on_train_start(self, *, split: Split, **kwargs) -> None:
        """
        Checks the DataFrame of the training and validation datasets.
        """
        self._check_df(split.train_dataset.df)
        self._check_df(split.val_dataset.df)

    def on_test_start(
        self,
        *,
        dataloader: DataLoader,
        **kwargs,
    ) -> None:
        """
        Checks the DataFrame of the test dataset.
        """
        self._check_df(dataloader.dataset.df)

    def on_predict_start(
        self,
        *,
        dataloader: DataLoader,
        **kwargs,
    ) -> None:
        """
        Checks the DataFrame of the prediction dataset.
        """
        self._check_df(dataloader.dataset.df)

    @staticmethod
    def _check_df(df: pd.DataFrame):
        """
        Checks that "participant_id" and "session_id" are in the DataFrame.
        """
        try:
            df = read_data(df, check_protected_names=False, check_duplicates=False)
        except DataFrameError as e:
            raise DataFrameError(
                "The DataFrame of your clinicadl.data.dataset.Dataset is not valid."
            ) from e


class _CheckDataLeakage:
    """
    Checks data leakage, i.e. that evaluation subjects were not seen during training.
    """

    def on_train_start(self, *, split: Split, **kwargs) -> None:
        """
        Checks leakage between training and validation.
        """
        common_subjects = self._get_common_subjects(
            split.train_dataset.df, split.val_dataset.df
        )
        if len(common_subjects) > 0:
            raise DataLeakageError(
                f"Some participants are in the training and validation sets: {common_subjects}"
            )

    def on_test_start(self, *, maps: Maps, dataloader: DataLoader, **kwargs) -> None:
        """
        Checks leakage between test and training+validation.
        """
        common_subjects = self._get_common_subjects(
            maps.open_file(maps.training.data.data_tsv), dataloader.dataset.df
        )
        if len(common_subjects) > 0:
            raise DataLeakageError(
                f"Some test participants are in the training/validation participants: {common_subjects} (see: {str(maps.training.data.data_tsv)})"
            )

    @staticmethod
    def _get_common_subjects(*dfs: pd.DataFrame) -> set[str]:
        """
        Gets subjects that are in all the input DataFrames.
        """
        return set.intersection(*(set(df[PARTICIPANT_ID]) for df in dfs))
