from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from clinicadl.metrics.handler import MetricsHandler
from clinicadl.utils.dictionary.words import PARTICIPANT_ID, SESSION_ID
from clinicadl.utils.objects import to_json_safe

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import DataLoader
    from clinicadl.io import Maps
    from clinicadl.io.maps.inference import InferenceDirType
    from clinicadl.models import Model
    from clinicadl.optim import OptimizationConfig
    from clinicadl.split import Split
    from clinicadl.train import ComputationalConfig

    from ..handler import CallbacksHandler


class ConfigSaverCallback(Callback):
    """
    Saves the configuration files and the (participant, session)
    pairs in the data.
    """

    def on_trainer_init(
        self,
        *,
        model: Model,
        maps: Maps,
        metrics: MetricsHandler,
        optimization: OptimizationConfig,
        callbacks: CallbacksHandler,
        **kwargs,
    ) -> None:
        if not maps.model_json.exists():
            to_json_safe(model, maps.model_json)
        if not maps.metrics_json.exists():
            metrics.to_json(maps.metrics_json)
        if not maps.training.optimization_json.exists():
            optimization.to_json(maps.training.optimization_json)
        if not maps.callbacks_json.exists():
            callbacks.to_json(maps.callbacks_json)

    def on_train_start(
        self,
        *,
        maps: Maps,
        split: Split,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        to_json_safe(
            split.train_dataset,
            maps.training.data.train.splits[split.index].dataset_json,
        )
        to_json_safe(
            split.val_dataset,
            maps.training.data.validation.splits[split.index].dataset_json,
        )

        split.config.train_loader_config.to_json(
            maps.training.data.train.splits[split.index].dataloader_json
        )
        split.config.val_loader_config.to_json(
            maps.training.data.validation.splits[split.index].dataloader_json
        )

        train_df = _clean_df(split.train_dataset.df)
        val_df = _clean_df(split.val_dataset.df)
        _update_training_df(maps, df=_join_dfs(train_df, val_df))
        maps.save_file(
            train_df,
            maps.training.data.train.splits[split.index].data_tsv,
        )
        maps.save_file(
            val_df,
            maps.training.data.validation.splits[split.index].data_tsv,
        )

        computational.to_json(maps.training.splits[split.index].computational_json)

    def on_test_start(
        self,
        *,
        maps: Maps,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        self._on_inference_start(
            maps,
            maps.test,
            dataloader,
            model_checkpoint,
            group_name,
            computational,
        )

    def on_predict_start(
        self,
        *,
        maps: Maps,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        self._on_inference_start(
            maps,
            maps.prediction,
            dataloader,
            model_checkpoint,
            group_name,
            computational,
        )

    def _on_inference_start(
        self,
        maps: Maps,
        inference_dir: InferenceDirType,
        dataloader: DataLoader,
        model_checkpoint: str,
        group_name: str,
        computational: ComputationalConfig,
    ):
        group_dir = inference_dir.groups[group_name]
        if not group_dir.dataset_json.exists():
            to_json_safe(
                dataloader.dataset,
                group_dir.dataset_json,
            )
        if not group_dir.dataloader_json.exists():
            dataloader.to_json(group_dir.dataloader_json)
        if not group_dir.data_tsv.exists():
            df = _clean_df(dataloader.dataset.df)
            maps.save_file(
                df,
                inference_dir.groups[group_name].data_tsv,
            )

        split_idx, chkpt = maps.training.read_checkpoint_name(model_checkpoint)
        computational.to_json(
            inference_dir.groups[group_name]
            .results.splits[split_idx]
            .models[chkpt]
            .computational_json
        )


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the columns "participant_id" and "session_id" and remove duplicated.
    """
    return (
        df[[PARTICIPANT_ID, SESSION_ID]]
        .drop_duplicates()
        .sort_values([PARTICIPANT_ID, SESSION_ID])
    )


def _join_dfs(*df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins two DataFrames, drops duplicates and sorts.
    """
    return pd.concat(df).drop_duplicates().sort_values([PARTICIPANT_ID, SESSION_ID])


def _update_training_df(maps: Maps, df: pd.DataFrame) -> None:
    """
    Updates the DataFrame containing all the training data.
    """
    tsv_path = maps.training.data.data_tsv
    if tsv_path.is_file():
        old_df = maps.open_file(tsv_path)
    else:
        old_df = pd.DataFrame()

    maps.save_file(_join_dfs(old_df, df), tsv_path, overwrite=True)
