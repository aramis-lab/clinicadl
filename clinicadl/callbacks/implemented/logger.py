from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from tqdm import tqdm

from clinicadl.io.maps import MapsSummary
from clinicadl.io.maps.training import TrainingSummary
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import CLINICADL
from clinicadl.utils.enum import TrainerCall, TrainerStage
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import BatchType
    from clinicadl.io import Maps
    from clinicadl.io.maps.exec import RunDir
    from clinicadl.losses.types import LossType
    from clinicadl.models import Model
    from clinicadl.split import Split
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
    To configure logging.

    `Logging <https://docs.python.org/3/library/logging.html>`_ is a convenient way to get insight of the current status of
    the :py:class:`~clinicadl.train.Trainer`, to record the important information
    that it raises, and to get clues on how to debug a failed execution.

    .. note::
        Some messages logged during the setup phase of :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`
        (or :py:meth:`validate <clinicadl.train.Trainer.train>`, etc.) may not be handled by this callback, because
        this callback is activated after that setup phase.

    Parameters
    ----------
    save_logs: bool, default=True
        Whether log messages raised during the code execution should be saved in files. Otherwise, log messages
        will just be printed in your console.

        These files will be saved in your :term:`MAPS` directory in ``<maps>/exec/run-<program_executed>_<datetime>``.
        In this folder, you will find:

            - ``debug.log`` (if ``debug=True``): log messages with level "DEBUG". Useful to debug your execution when it failed.
            - ``info.log``: log messages with level "INFO". General information about the execution flow.
            - ``error.log``: log messages with level "ERROR". To have information on potential errors that stopped the execution.

        Note that a file named ``warning.log``, which contain log messages with level "WARNING", will also be saved,
        but its location depends on the program being executed. For example, if you execute :py:meth:`Trainer.train <clinicadl.train.Trainer.train>`,
        it will be saved under ``<maps>/training/split-<split_idx>/warning.log``. This is because this file may contain
        methodological warnings that should be stayed close to the results.

    debug: bool, default=True
        Whether to print/save the log messages with level "DEBUG". These messages are verbose, but helpful for debugging.

    progress_bar : bool, default=True
        Whether to display a progress bar every time an iteration on a dataloader is performed.
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
        self._summary: Optional[MapsSummary] = None
        self._train_summary: Optional[TrainingSummary] = None
        self._train_progress_bar: Optional[tqdm] = None
        self._val_progress_bar: Optional[tqdm] = None
        self._test_progress_bar: Optional[tqdm] = None
        self._predict_progress_bar: Optional[tqdm] = None
        self._output_path: Optional[Path] = None
        self._log_path: Optional[Path] = None

    def on_trainer_init(
        self,
        *,
        model: Model,
        maps: Maps,
        **kwargs,
    ) -> None:
        if not maps.architecture_log.is_file():
            maps.save_file(repr(model), maps.architecture_log)
        self._summary = MapsSummary(maps.summary_log)

    def on_exception(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        if state.called == TrainerCall.TRAIN and self._train_summary:
            self._train_summary.add_training_end_info(
                n_epochs=state.current_epoch, interrupted=True
            )
        if self.config.save_logs and self.logger:
            self.logger.error(
                "An exception occurred. To debug, check the logs in %s", self._log_path
            )
            _shutdown_logging(self.logger)

    def on_train_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        split: Split,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        split_dir = maps.training.splits[split.index]
        self._setup_logging(maps, state, warning_file=split_dir.warning_log)

        self.logger.info("Beginning of training on split %s", split.index)
        self.logger.info("Computational configuration: %s", computational)

        self._output_path = split_dir.path

        self._train_summary = TrainingSummary(
            maps.training.splits[split.index].summary_log
        )
        self._train_summary.create()
        self._train_summary.add_data_info(
            n_train_samples=len(split.train_dataset),
            n_val_samples=len(split.val_dataset),
        )
        self._summary.add_training_split(split.index)

    def on_resume(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        split: Split,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        split_dir = maps.training.splits[split.index]
        self._setup_logging(maps, state, warning_file=split_dir.warning_log)

        last_epoch = sorted(maps.training.splits[split.index].tmp.epochs_list)[-1]
        self.logger.info(
            "Resuming training on split %s from epoch %d", split.index, last_epoch + 1
        )
        self.logger.info("Computational configuration: %s", computational)

        self._output_path = split_dir.path
        self._train_summary = TrainingSummary(
            maps.training.splits[split.index].summary_log
        )

    def on_validate_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        model_checkpoint: str,
        **kwargs,
    ) -> None:
        model_dir = maps.training.splits[state.split_idx].models.get_checkpoint_dir(
            model_checkpoint
        )

        self._setup_logging(maps, state, warning_file=model_dir.warning_log)
        self._output_path = model_dir.path

        self.logger.info(
            "Beginning of validation of checkpoint '%s' on split %d",
            model_checkpoint,
            state.split_idx,
        )

        self._val_progress_bar = tqdm(
            total=state.num_val_batches,
            unit="batch",
            desc="Validation",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
            dynamic_ncols=True,
        )

    def on_validation_start(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        self._train_progress_bar.close()
        self.logger.info("Beginning of validation")

        self._val_progress_bar = tqdm(
            total=state.num_val_batches,
            unit="batch",
            desc="Validation",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
            dynamic_ncols=True,
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
        chkpt_split, chkpt_name = maps.training.read_checkpoint_name(model_checkpoint)
        model_dir = (
            maps.test.groups[group_name].results.splits[chkpt_split].models[chkpt_name]
        )
        self._setup_logging(maps, state, warning_file=model_dir.warning_log)

        self.logger.info("Beginning of test")

        self._output_path = model_dir.path

        self._test_progress_bar = tqdm(
            total=state.num_test_batches,
            unit="batch",
            desc="Test",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
            dynamic_ncols=True,
        )

        self._summary.add_test_group(group_name)

    def on_predict_start(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        model_checkpoint: str,
        group_name: str,
        **kwargs,
    ) -> None:
        chkpt_split, chkpt_name = maps.training.read_checkpoint_name(model_checkpoint)
        model_dir = (
            maps.prediction.groups[group_name]
            .results.splits[chkpt_split]
            .models[chkpt_name]
        )
        self._setup_logging(maps, state, warning_file=model_dir.warning_log)

        self.logger.info("Beginning of prediction")

        self._output_path = model_dir.path

        self._predict_progress_bar = tqdm(
            total=state.num_pred_batches,
            unit="batch",
            desc="Prediction",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
            dynamic_ncols=True,
        )

        self._summary.add_prediction_group(group_name)

    def on_train_end(
        self,
        *,
        state: TrainerState,
        **kwargs,
    ) -> None:
        self._train_summary.add_training_end_info(
            n_epochs=state.current_epoch, interrupted=False
        )
        self.logger.info(
            "Training completed successfully (stopped after %s epochs)",
            state.current_epoch,
        )
        self.logger.info(
            "All results, logs, and model checkpoints are saved in %s",
            self._output_path,
        )
        _shutdown_logging(self.logger)

    def on_validate_end(
        self,
        **kwargs,
    ) -> None:
        self.on_validation_end()

        self.logger.info("Validation metrics saved in %s", self._output_path)
        _shutdown_logging(self.logger)

    def on_validation_end(
        self,
        **kwargs,
    ) -> None:
        self._val_progress_bar.close()

        self.logger.info("End of validation")

    def on_test_end(
        self,
        **kwargs,
    ) -> None:
        self._test_progress_bar.close()

        self.logger.info("End of test")
        self.logger.info("Test metrics saved in %s", self._output_path)
        _shutdown_logging(self.logger)

    def on_predict_end(
        self,
        **kwargs,
    ) -> None:
        self._predict_progress_bar.close()

        self.logger.info("End of prediction")
        self.logger.info("Predictions saved in %s", self._output_path)
        _shutdown_logging(self.logger)

    def on_epoch_start(self, *, state: TrainerState, **kwargs) -> None:
        self.logger.info("Beginning of epoch %d", state.current_epoch)

        self._train_progress_bar = tqdm(
            total=state.num_train_batches,
            unit="batch",
            desc=f"Epoch {state.current_epoch}/{state.num_epochs}",
            initial=1,
            disable=not self.config.progress_bar,
            file=sys.stdout,
            dynamic_ncols=True,
        )

    def on_epoch_end(self, *, state: TrainerState, **kwargs) -> None:
        self._train_progress_bar.close()
        self.logger.info("Epoch %d completed", state.current_epoch)

    def on_batch_start(
        self,
        *,
        model: Model,
        maps: Maps,
        state: TrainerState,
        batch: BatchType,
        **kwargs,
    ) -> None:
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

        if not maps.nn_summary_txt.is_file():
            with torch.no_grad():
                nn_summary = model.get_summary(batch)
            maps.save_file(nn_summary, maps.nn_summary_txt)

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

    def _setup_logging(
        self, maps: Maps, state: TrainerState, warning_file: Path
    ) -> None:
        self._log_path = _get_log_file_dir(maps, state, self.config.save_logs)
        self.logger = _setup_logging(
            self.config.debug,
            warning_file=warning_file,
            log_directory=self._log_path,
        )


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


def _setup_logging(
    debug: bool, warning_file: Path, log_directory: Optional[RunDir]
) -> logging.Logger:
    """
    Setup ClinicaDL's logging facilities.
    """
    logger = logging.getLogger(CLINICADL)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers = []  # clear existing handlers
    if log_directory:
        log_directory.create(exist_ok=True)

    datefmt = "%Y-%m-%d %H:%M:%S"

    # Standard output handler (INFO, WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s: %(message)s", datefmt=datefmt
    )

    outputs_handler = logging.StreamHandler(stream=sys.stdout)
    outputs_handler.setLevel(logging.INFO)
    outputs_handler.addFilter(_LogsFilter())
    outputs_handler.setFormatter(formatter)
    logger.addHandler(outputs_handler)

    if log_directory:
        info_handler = logging.FileHandler(
            log_directory.info_log, mode="a", encoding="utf-8"
        )
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(_LogsFilter())
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        warning_file_handler = logging.FileHandler(
            warning_file, mode="a", encoding="utf-8"
        )
        warning_file_handler.setLevel(logging.WARNING)
        warning_file_handler.addFilter(_LogsFilter(level=logging.WARNING))
        warning_file_handler.setFormatter(formatter)
        logger.addHandler(warning_file_handler)

        # DEBUG
        if debug:
            debug_file_handler = logging.FileHandler(
                log_directory.debug_log, mode="a", encoding="utf-8"
            )
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.addFilter(_LogsFilter(level=logging.DEBUG))
            debug_file_handler.setFormatter(formatter)
            logger.addHandler(debug_file_handler)

    # Standard error handler (ERROR and above)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: %(message)s", datefmt=datefmt
    )

    error_handler = logging.StreamHandler(stream=sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    if log_directory:
        error_file_handler = logging.FileHandler(
            log_directory.error_log, mode="a", encoding="utf-8"
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        logger.addHandler(error_file_handler)

    return logger


def _shutdown_logging(logger: logging.Logger) -> None:
    """
    To close ClinicaDL's logging facilities and come back to normal logging.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


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
