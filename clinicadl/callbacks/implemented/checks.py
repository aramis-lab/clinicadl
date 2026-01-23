from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, TypeVar, Union

import pandas as pd
import torch

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.dataloader.config import get_dataloader_from_json_safely
from clinicadl.data.datasets.factory import get_dataset_from_json_safely
from clinicadl.utils.dictionary.words import PARTICIPANT_ID, SESSION_ID
from clinicadl.utils.exceptions import (
    DataFrameError,
    DataLeakageError,
)
from clinicadl.utils.tsvtools import read_data

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import DataLoader
    from clinicadl.data.datasets import Dataset
    from clinicadl.data.datasets.base import BaseDataset
    from clinicadl.io import Maps
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.split import Split
    from clinicadl.train import TrainerState

logger = logging.getLogger("clinicadl.callbacks.implemented.ChecksCallback")
T = TypeVar("T")


class ChecksCallback(Callback):
    """
    Performs various checks on the objects defined by the user.
    """

    def __init__(self):
        self._check_inputs = _CheckInputs()
        self._check_dataframes = _CheckDataFrames()
        self._check_data_leakage = _CheckDataLeakage()
        self._check_losses = _CheckLosses()
        self._check_data_consistency = _CheckDataConsistency()

    def on_train_start(self, **kwargs) -> None:
        self._check_inputs.on_train_start(**kwargs)
        self._check_losses.on_train_start(**kwargs)
        self._check_dataframes.on_train_start(**kwargs)
        self._check_data_leakage.on_train_start(**kwargs)
        self._check_data_consistency.on_train_start(**kwargs)

    def on_backward_step_start(self, **kwargs) -> None:
        self._check_losses.on_backward_step_start(**kwargs)

    def on_validate_start(self, **kwargs) -> None:
        self._check_data_consistency.on_validate_start(**kwargs)

    def on_test_start(self, **kwargs) -> None:
        self._check_inputs.on_test_start(**kwargs)
        self._check_dataframes.on_test_start(**kwargs)
        self._check_data_leakage.on_test_start(**kwargs)
        self._check_data_consistency.on_test_start(**kwargs)

    def on_predict_start(self, **kwargs) -> None:
        self._check_inputs.on_predict_start(**kwargs)
        self._check_dataframes.on_predict_start(**kwargs)
        self._check_data_consistency.on_predict_start(**kwargs)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


class _CheckInputs:
    """
    Various checks on user inputs.
    """

    def on_train_start(self, *, maps: Maps, split: Split, **kwargs) -> None:
        """
        Checks the split index and that dataloaders have been instantiated in the splits.
        """
        if split.index in maps.training.splits_list:
            raise ValueError(
                f"Training on split {split.index}. To relaunch a training on this split, first delete it properly with clinicadl.io.Maps.delete_split"
            )

        if split.train_loader is None:
            raise RuntimeError(
                "The split has no training dataloder defined. Please run 'build_train_loader'"
            )
        if split.val_loader is None:
            raise RuntimeError(
                "The split has no validation dataloder defined. Please run 'build_val_loader'"
            )

    def on_test_start(
        self,
        *,
        maps: Maps,
        group_name: str,
        model_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Checks if the checkpoint has already been tested on this group.
        """
        self._check_checkpoint(model_checkpoint, group_name, maps=maps, test=True)

    def on_predict_start(
        self,
        *,
        maps: Maps,
        group_name: str,
        model_checkpoint: str,
        **kwargs,
    ) -> None:
        """
        Checks if the group has already been predicted with this checkpoint.
        """
        self._check_checkpoint(model_checkpoint, group_name, maps=maps, test=False)

    @staticmethod
    def _check_checkpoint(
        model_checkpoint: str, group_name: str, maps: Maps, test: bool
    ):
        split_idx, chkpt = maps.training.read_checkpoint_name(model_checkpoint)
        if test:
            dir_ = maps.test
        else:
            dir_ = maps.prediction

        if chkpt in dir_.groups[group_name].results.splits[split_idx].models_list:
            raise FileExistsError(
                f"There are already some results for checkpoint '{chkpt}' in {dir_.groups[group_name].results.splits[split_idx].path}. "
                f"Set overwrite=True in Trainer.{'test' if test else 'predict'} to overwrite them."
            )


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
                f"Some participants are in the training and validation sets: {sorted(list(common_subjects))}"
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
                f"Some test participants are in the training/validation participants: {sorted(list(common_subjects))} (see: {str(maps.training.data.data_tsv)})"
            )

    @staticmethod
    def _get_common_subjects(*dfs: pd.DataFrame) -> set[str]:
        """
        Gets subjects that are in all the input DataFrames.
        """
        return set.intersection(*(set(df[PARTICIPANT_ID]) for df in dfs))


class _CheckDataConsistency:
    def on_train_start(self, *, maps: Maps, split: Split, **kwargs) -> None:
        """
        Compares:
        - training datasets and dataloaders with previous splits;
        - validation dataset with previous splits.

        Raises warnings if these objects differ or cannot be compared.
        """
        for split_idx in maps.training.splits_list:
            self._compare(
                split.train_dataset,
                maps.training.data.train.splits[split_idx].dataset_json,
                getter=get_dataset_from_json_safely,
                comparator=_compare_datasets,
                new_group=split.index,
                old_group=split_idx,
                stage="training",
            )
            self._compare(
                split.val_dataset,
                maps.training.data.validation.splits[split_idx].dataset_json,
                getter=get_dataset_from_json_safely,
                comparator=_compare_datasets,
                new_group=split.index,
                old_group=split_idx,
                stage="validation",
            )
            self._compare(
                split.config.train_loader_config,
                maps.training.data.train.splits[split_idx].dataloader_json,
                getter=get_dataloader_from_json_safely,
                comparator=_compare_dataloaders,
                new_group=split.index,
                old_group=split_idx,
                stage="training",
            )

    def on_validate_start(
        self, *, maps: Maps, state: TrainerState, dataloader: DataLoader, **kwargs
    ) -> None:
        """
        Raises:
        - error if the (participant, session) couples in the dataset are not the same as the validation couples;
        - warning if the dataset doesn't match with the original validation dataset (or cannot be compared).
        """
        self._compare_participants_sessions(
            dataloader.dataset,
            maps.training.data.validation.splits[state.split_idx].data_tsv,
            maps=maps,
            stage="validation",
            group=state.split_idx,
        )

        self._compare(
            dataloader.dataset,
            maps.training.data.validation.splits[state.split_idx].dataset_json,
            getter=get_dataset_from_json_safely,
            comparator=_compare_datasets,
            new_group=state.split_idx,
            old_group=state.split_idx,
            stage="validate",
        )

    def on_test_start(
        self,
        *,
        maps: Maps,
        dataloader: DataLoader,
        group_name: str,
        **kwargs,
    ) -> None:
        """
        Raises:
        - error if the (participant, session) couples in the dataset are not the same as the group couples;
        - warning if the dataset doesn't match with the original test dataset (or cannot be compared).
        """
        self._compare_participants_sessions(
            dataloader.dataset,
            maps.test.groups[group_name].data_tsv,
            maps=maps,
            stage="test",
            group=group_name,
        )

        self._compare(
            dataloader.dataset,
            maps.test.groups[group_name].dataset_json,
            getter=get_dataset_from_json_safely,
            comparator=_compare_datasets,
            new_group=group_name,
            old_group=group_name,
            stage="test",
        )

    def on_predict_start(
        self,
        *,
        maps: Maps,
        dataloader: DataLoader,
        group_name: str,
        **kwargs,
    ) -> None:
        """
        Raises:
        - error if the (participant, session) couples in the dataset are not the same as the group couples;
        - warning if the dataset doesn't match with the original prediction dataset (or cannot be compared).
        """
        self._compare_participants_sessions(
            dataloader.dataset,
            maps.prediction.groups[group_name].data_tsv,
            maps=maps,
            stage="prediction",
            group=group_name,
        )

        self._compare(
            dataloader.dataset,
            maps.prediction.groups[group_name].dataset_json,
            getter=get_dataset_from_json_safely,
            comparator=_compare_datasets,
            new_group=group_name,
            old_group=group_name,
            stage="prediction",
        )

    @staticmethod
    def _compare_participants_sessions(
        new: Dataset,
        old_tsv_path: Path,
        maps: Maps,
        stage: str,
        group: Union[str, int],
    ) -> None:
        """
        Compares (participant, session) pairs in a dataset with the ones in a tsv file.
        """
        old_participants_sessions = maps.open_file(old_tsv_path)
        diff = new.get_participant_session_couples().symmetric_difference(
            set(
                zip(
                    old_participants_sessions[PARTICIPANT_ID],
                    old_participants_sessions[SESSION_ID],
                )
            )
        )
        if diff:
            ref_group = f"split-{group}" if isinstance(group, int) else f"group-{group}"
            raise ValueError(
                f"The {stage} dataset passed does not contain the same (participant, session) pairs "
                f"as in the original {stage} dataset of {ref_group} (in {old_tsv_path}). Difference: {sorted(list(diff))}",
            )

    @staticmethod
    def _compare(
        new: T,
        old_path: Path,
        getter: Callable[[Path, T], tuple[Optional[T], list[str]]],
        comparator: Callable[[T, T, list[str]], Optional[str]],
        new_group: Union[str, int],
        old_group: Union[str, int],
        stage: str,
    ) -> None:
        """
        Compares a dataset (or dataloader) with one serialized in a file.
        Raises warning if different or if the file cannot be read properly.
        """
        args = {
            "phase": stage.replace("validate", "validation"),
            "type": "dataloader" if isinstance(new, DataLoaderConfig) else "dataset",
            "new_group": str(new_group),
            "old_group": str(old_group),
            "group_type": "split" if isinstance(new_group, int) else "group",
            "old_path": old_path,
            "compared_with": f"of split-{new_group}"
            if stage == "training" or stage == "validation"
            else f"passed to Trainer.{stage.replace('ion', '')}",
            "across": "across splits"
            if stage == "training" or stage == "validation"
            else "in predictions"
            if stage == "prediction"
            else f"in {stage.replace('validate', 'validation')} metrics",
        }

        old, problematic_fields = getter(
            old_path,
            default=new,
        )
        if not old:
            logging.warning(
                "Could not read the %(phase)s %(type)s of %(group_type)s-%(old_group)s (in %(old_path)s), and thus could not compare with the %(type)s %(compared_with)s. "
                "Beware that differences between %(type)ss could lead to inconsistent results %(across)s.",
                args,
            )
            return
        if problematic_fields:
            args["problematic_fields"] = problematic_fields
            logging.warning(
                "Could not read the arguments %(problematic_fields)s of the %(phase)s %(type)s of %(group_type)s-%(old_group)s (in %(old_path)s), and thus could not compare with the %(type)s %(compared_with)s. "
                "Beware that differences between %(type)ss could lead to inconsistent results %(across)s.",
                args,
            )
        if error_msg := comparator(new, old, problematic_fields):
            args["error_msg"] = error_msg
            logging.warning(
                "The %(phase)s %(type)ss of %(group_type)s-%(old_group)s and the one %(compared_with)s are different: %(error_msg)s\nThis may lead to inconsistent results %(across)s.",
                args,
            )


def _compare_dataloaders(
    new: DataLoaderConfig,
    old: DataLoaderConfig,
    except_fields: list[str],
) -> Optional[str]:
    """
    To compare two dataloaders. Returns potential differences in a message.
    """
    if "batch_size" not in except_fields and new.batch_size != old.batch_size:
        return f"the two dataloaders have different batch sizes. Got {new.batch_size} and {old.batch_size}"
    if (
        "sampling_weights" not in except_fields
        and new.sampling_weights != old.sampling_weights
    ):
        return f"the two dataloaders have different sampling weight columns. Got {new.sampling_weights} and {old.sampling_weights}"
    if "shuffle" not in except_fields and new.shuffle != old.shuffle:
        return "one dataloader is shuffled but the other is not."
    if "drop_last" not in except_fields and new.drop_last != old.drop_last:
        return (
            "one dataloader has drop_last=True whereas the other has drop_last=False."
        )
    if "collate_fn" not in except_fields and new.collate_fn != old.collate_fn:
        return f"the two dataloaders have different collate function. Got {new.collate_fn} and {old.collate_fn}"


def _compare_datasets(
    new: Dataset, old: BaseDataset, except_fields: list[str]
) -> Optional[str]:
    """
    To compare two datasets. Returns potential differences in a message.
    """
    if type(new) is not type(old):
        return (
            f"the two datasets are not the same type. Got {type(new)} and {type(old)}"
        )
    if (
        "directory" not in except_fields
        and new.config.directory != old.config.directory
    ):
        return f"the two datasets don't come from the same directory: {str(new.config.directory)} and {str(old.config.directory)}"
    if "datatype" not in except_fields and new.config.datatype != old.config.datatype:
        return f"the two datasets don't have the same datatypes, which differ in their pattern or key. Got {new.config.datatype} and {old.config.datatype}"
    if "label" not in except_fields and new.config.label != old.config.label:
        return f"the two datasets don't have the same label. Got {new.config.label} and {old.config.label}"
    if (
        "transforms" not in except_fields
        and new.config.transforms != old.config.transforms
    ):
        return f"the two datasets don't have the same transforms. Got {new.config.transforms}\nand\n\n{old.config.transforms}"
    if "masks" not in except_fields and (masks_1 := set(new.config.masks)) != (
        masks_2 := set(old.config.masks)
    ):
        return (
            f"the two datasets don't have the same masks. Got {masks_1} and {masks_2}"
        )
    if "columns" not in except_fields and (columns_1 := set(new.config.columns)) != (
        columns_2 := set(old.config.columns)
    ):
        return f"the two datasets don't have the same columns or column processing. Got {columns_1} and {columns_2}"
