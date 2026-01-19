from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

from clinicadl.metrics.handler import MetricsHandler
from clinicadl.utils.dictionary.words import PARTICIPANT_ID, SESSION_ID
from clinicadl.utils.exceptions import DataFrameError
from clinicadl.utils.objects import to_json_safe
from clinicadl.utils.tsvtools import read_data

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import DataLoader
    from clinicadl.io import Maps
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
        to_json_safe(model, maps.model_json)
        metrics.to_json(maps.metrics_json)
        optimization.to_json(maps.training.optimization_json)
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
        group_name: str,
        **kwargs,
    ) -> None:
        if group_name not in maps.test.groups_list:
            maps.test.create_group(group_name)
            to_json_safe(
                dataloader.dataset,
                maps.test.groups[group_name].dataset_json,
            )
            df = _clean_df(dataloader.dataset.df)
            maps.save_file(
                df,
                maps.test.groups[group_name].data_tsv,
            )

    def on_predict_start(
        self,
        *,
        maps: Maps,
        dataloader: DataLoader,
        group_name: str,
        **kwargs,
    ) -> None:
        if group_name not in maps.prediction.groups_list:
            maps.prediction.create_group(group_name)
            to_json_safe(
                dataloader.dataset,
                maps.prediction.groups[group_name].dataset_json,
            )
            df = _clean_df(dataloader.dataset.df)
            maps.save_file(
                df,
                maps.prediction.groups[group_name].data_tsv,
            )

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only the columns "participant_id" and "session_id" and remove duplicated.
    """
    try:
        df = read_data(df, check_protected_names=False, check_duplicates=False)
    except DataFrameError as e:
        raise DataFrameError(
            "The DataFrame of your clinicadl.data.dataset.Dataset is not valid."
        ) from e
    df = df[[PARTICIPANT_ID, SESSION_ID]]

    return df.drop_duplicates().sort_values([PARTICIPANT_ID, SESSION_ID])


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
