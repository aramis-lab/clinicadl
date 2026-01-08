from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from tqdm import tqdm

from clinicadl.train.trainer_state import TrainerCall, TrainerStage
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.io.maps.exec import RunDir
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.train import TrainerState
    from clinicadl.train.computational import ComputationalConfig


class LoggerCallbackConfig(ObjectConfig["LoggerCallback"]):
    """Config class for ``LoggerCallback``."""

    save_logs: bool
    debug: bool
    progress_bar: bool

    @classmethod
    def _get_class(cls):
        return LoggerCallback


class LoggerCallback(Callback, HasConfig[LoggerCallbackConfig]):
    """
    Callback that logs major training events to console and/or file.

    Parameters
    ----------
    verbose : bool, default=False
        If True, enables detailed DEBUG-level logging.
    """

    _config_type = LoggerCallbackConfig

    def __init__(
        self,
        save_logs: bool = True,
        debug: bool = True,
        progress_bar: bool = True,
    ):
        self.config = self._config_type(
            save_logs=save_logs, debug=debug, progress_bar=progress_bar
        )
        self.logger: Optional[logging.Logger] = None
        self._train_progress_bar: Optional[tqdm] = None
        self._val_progress_bar: Optional[tqdm] = None
        self._test_progress_bar: Optional[tqdm] = None
        self._predict_progress_bar: Optional[tqdm] = None
        self._output_path: Optional[Path]

    def on_train_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        self.logger = _setup_logging(
            self.config.debug,
            log_directory=_get_log_file_dir(maps, state, self.config.save_logs),
        )
        self._output_path = maps.training.splits[state.split_idx].path

        self.logger.info("Beginning of training on split %s", state.split_idx)
        self.logger.info("Computational configuration: %s", computational)

    def on_validation_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        model_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> None:
        if state.called == TrainerCall.VALIDATE:
            self.logger = _setup_logging(
                self.config.debug,
                log_directory=_get_log_file_dir(maps, state, self.config.save_logs),
            )

            if model_checkpoint:
                self._output_path = (
                    maps.training.splits[state.split_idx]
                    .models.get_checkpoint_dir(model_checkpoint)
                    .path
                )
            else:
                self._output_path = maps.training.splits[state.split_idx].models.path

        self.logger.info("Beginning of validation")

        self._val_progress_bar = tqdm(
            total=state.num_val_batches,
            unit="batch",
            desc="Validation",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
        )

    def on_test_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        model_checkpoint: str,
        group_name: str,
        **kwargs,
    ) -> None:
        self.logger = _setup_logging(
            self.config.debug,
            log_directory=_get_log_file_dir(maps, state, self.config.save_logs),
        )

        self.logger.info("Beginning of test")

        chkpt_split, chkpt_name = maps.training.read_checkpoint_name(model_checkpoint)
        self._output_path = (
            maps.test.groups[group_name].results.splits[chkpt_split].models[chkpt_name]
        ).path

        self._test_progress_bar = tqdm(
            total=state.num_test_batches,
            unit="batch",
            desc="Test",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
        )

    def on_predict_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        model_checkpoint: str,
        group_name: str,
        **kwargs,
    ) -> None:
        self.logger = _setup_logging(
            self.config.debug,
            log_directory=_get_log_file_dir(maps, state, self.config.save_logs),
        )

        self.logger.info("Beginning of prediction")

        chkpt_split, chkpt_name = maps.training.read_checkpoint_name(model_checkpoint)
        self._output_path = (
            maps.prediction.groups[group_name]
            .results.splits[chkpt_split]
            .models[chkpt_name]
        ).path

        self._predict_progress_bar = tqdm(
            total=state.num_pred_batches,
            unit="batch",
            desc="Prediction",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
        )

    def on_train_end(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        self.logger.info(
            "Training completed successfully (stopped after %s epochs)",
            state.current_epoch,
        )
        self.logger.info(
            "All results, logs, and model checkpoints are saved in %s",
            self._output_path,
        )

    def on_validation_end(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        self._val_progress_bar.close()

        self.logger.info("End of validation")
        if state.called == TrainerCall.VALIDATE:
            self.logger.info("Validation metrics saved in %s", self._output_path)

    def on_test_end(
        self,
        **kwargs,
    ) -> None:
        self._test_progress_bar.close()

        self.logger.info("End of test")
        self.logger.info("Test metrics saved in %s", self._output_path)

    def on_predict_end(
        self,
        **kwargs,
    ) -> None:
        self._predict_progress_bar.close()

        self.logger.info("End of prediction")
        self.logger.info("Predictions saved in %s", self._output_path)

    def on_epoch_start(self, *, state: TrainerState, **kwargs) -> None:
        self.logger.info("Beginning of epoch %d", state.current_epoch)

        self._train_progress_bar = tqdm(
            total=state.num_train_batches,
            unit="batch",
            desc=f"Epoch {state.current_epoch}/{state.num_epochs}",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
        )

    def on_epoch_end(self, *, state: TrainerState, **kwargs) -> None:
        self._train_progress_bar.close()
        self.logger.info("Epoch %d completed", state.current_epoch)

    def on_batch_start(self, *, state: TrainerState, **kwargs) -> None:
        if state.stage == TrainerStage.TRAIN:
            current_batch = state.current_train_batch

        elif state.stage == TrainerStage.PRED:
            current_batch = state.current_pred_batch

        elif state.called == TrainerCall.TEST:
            current_batch = state.current_test_batch

        elif state.stage == TrainerStage.EVAL:
            current_batch = state.current_val_batch

        else:
            raise ValueError("Inconsistent 'stage' and 'called' in the TrainerState.")

        self.logger.debug("Batch %d loaded", current_batch)

    def on_batch_end(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        if state.stage == TrainerStage.TRAIN:
            current_batch = state.current_train_batch
            pbar = self._train_progress_bar

        elif state.stage == TrainerStage.PRED:
            current_batch = state.current_pred_batch
            pbar = self._predict_progress_bar

        elif state.called == TrainerCall.TEST:
            current_batch = state.current_test_batch
            pbar = self._test_progress_bar

        elif state.stage == TrainerStage.EVAL:
            current_batch = state.current_val_batch
            pbar = self._val_progress_bar

        else:
            raise ValueError("Inconsistent 'stage' and 'called' in the TrainerState.")

        _update_progress_bar(pbar, n=current_batch)
        self.logger.debug("Processing of batch %d completed", current_batch)

    def on_backward_step_start(
        self,
        *,
        model: Model,
        loss: LossType,
        **kwargs,
    ) -> None:
        if self.config.progress_bar:
            if not isinstance(loss, dict):
                loss_name = list(model.get_loss_functions().keys())[0]
                loss = {loss_name: loss}

            loss = {name: tensor.item() for name, tensor in loss.items()}

            self._train_progress_bar.set_postfix(loss)

    def state_dict(self) -> Mapping[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        pass


class _LogsFilter(logging.Filter):
    """
    Logging filter to filter out errors or keep only logs from one level.
    """

    def __init__(self, level: Optional[logging.LogRecord] = None):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        if self.level:
            return record.levelno == self.level
        return record.levelno <= logging.ERROR


def _setup_logging(debug: bool, log_directory: Optional[RunDir]) -> logging.Logger:
    """
    Setup ClinicaDL's logging facilities.
    """
    logger = logging.getLogger("clinicadl")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers = []  # clear existing handlers
    if log_directory:
        log_directory.create(exist_ok=True)

    datefmt = "%Y-%m-%d %H:%M:%S"

    # Standard output handler (INFO, WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s", datefmt=datefmt
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(_LogsFilter())
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_directory:
        file_handler = logging.FileHandler(
            log_directory.outputs, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.addFilter(_LogsFilter())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # DEBUG
        if debug:
            debug_file_handler = logging.FileHandler(
                log_directory.debug, mode="a", encoding="utf-8"
            )
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.addFilter(_LogsFilter(level=logging.DEBUG))
            debug_file_handler.setFormatter(formatter)
            logger.addHandler(debug_file_handler)

    # Standard error handler (ERROR and above)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s", datefmt=datefmt
    )

    err_handler = logging.StreamHandler(stream=sys.stderr)
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    if log_directory:
        error_file_handler = logging.FileHandler(
            log_directory.errors, mode="a", encoding="utf-8"
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    return logger


def _get_log_file_dir(
    maps: Maps, state: TrainerState, save_logs: bool
) -> Optional[RunDir]:
    """
    Gets the right directory for the current execution.
    """
    if not save_logs:
        return None

    run_name = maps.exec.create_run(process_called=state.called)
    return maps.exec.runs[run_name]


def _update_progress_bar(pbar: tqdm, n: int) -> None:
    """
    To update a progress bar with current count value.
    """
    pbar.n = n
    pbar.refresh()
