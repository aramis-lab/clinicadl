from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback


class Tensorboard(Callback):
    """
    Callback to log metrics and optionally the model graph to TensorBoard during training.

    Attributes
    ----------
        log_dir : Optional[str]
            Directory where TensorBoard logs will be saved. If None, defaults to a logs folder within the split directory.
        log_model_graph : bool
            Whether to log the model graph once at the start of training.

    .. note:
        Metrics are logged at the end of each epoch using values from the training state.
        The model graph can be logged once after training begins if an example input is provided.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_model_graph: bool = False,
        example_input: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        log_dir : str or Path, optional
            Directory to save TensorBoard logs. Defaults to None.
        log_model_graph : bool, default False
            Whether to log the model graph once at the start of training.
        example_input : torch.Tensor, optional
            Example input tensor used to generate the model graph if `log_model_graph` is True.
        """
        self.log_dir = log_dir
        self.log_model_graph = log_model_graph
        self.example_input = example_input
        self.writer = None

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        """
        Initialize the SummaryWriter, setting up the log directory.

        Optionally logs the model graph if requested.
        """
        if self.log_dir is None:
            self.log_dir = config.maps.training.splits[
                config.split.index
            ].logs.tensorboard

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        if self.log_model_graph:
            if self.example_input is None:
                raise ValueError(
                    "example_input must be provided to log the model graph."
                )
            self.writer.add_graph(config.model.network, self.example_input)

    def on_epoch_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Log metrics from the current epoch to TensorBoard.

        Assumes `config.metrics.df` is a pandas DataFrame with metrics logged per epoch.
        """
        if self.writer is None:
            raise RuntimeError("TensorBoard SummaryWriter not initialized")

        df = config.metrics.df
        epoch = config.epoch

        if df is not None and not df.empty:
            for metric in df.columns:
                try:
                    value = df.at[epoch, metric]
                    if value is not None:
                        self.writer.add_scalar(metric, float(value), epoch)
                except KeyError:
                    # Metric not found for current epoch, skip logging
                    continue

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        """
        Close the SummaryWriter to flush all pending events.
        """
        if self.writer is not None:
            self.writer.close()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the callback to a dictionary representation.

        Returns
        -------
        dict
            Dictionary representation of the callback.
        """
        json_dict = super().to_dict()
        json_dict.update(
            {
                "log_dir": self.log_dir,
                "log_model_graph": self.log_model_graph,
                "example_input": self.example_input,
            }
        )
        return json_dict
