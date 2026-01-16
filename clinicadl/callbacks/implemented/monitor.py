from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from pydantic import NonNegativeInt

from clinicadl.io.maps.training import TrainingSummary
from clinicadl.train.trainer_state import TrainerCall, TrainerStage, TrainerState
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.utils import SEP
from clinicadl.utils.dictionary.words import GPU
from clinicadl.utils.objects import HasConfig

from ..base import Callback

if TYPE_CHECKING:
    from clinicadl.io import Maps
    from clinicadl.optim import OptimizationConfig
    from clinicadl.split import Split
    from clinicadl.train import ComputationalConfig

logger = logging.getLogger("clinicadl.callbacks.MonitorCallback")

TRAIN = "Training"
EPOCH = "Epoch"
TRAIN_LOOP = "Training loop"
TRAIN_LOAD = "Training data loading"
FORWARD = "Forward"
FORWARD_GPU = "Forward GPU (s)"
FORWARD_MEM = "Forward GPU max memory (MB)"
BACKWARD = "Backward"
BACKWARD_GPU = "Backward GPU (s)"
BACKWARD_MEM = "Backward GPU max memory (MB)"
OPT = "Optimization"
OPT_GPU = "Optimization GPU (s)"
OPT_MEM = "Optimization GPU max memory (MB)"
VAL = "Validation"
VAL_LOOP = "Validation loop"
VAL_LOAD = "Validation data loading"
EVAL = "Evaluation"
EVAL_GPU = "Evaluation GPU (s)"
EVAL_MEM = "Evaluation GPU max memory (MB)"

OOM = "CUDA out of memory"


class MonitorCallbackConfig(ObjectConfig["MonitorCallback"]):
    """Config class for ``MonitorCallback``."""

    num_measurements: NonNegativeInt
    warmup_iterations: NonNegativeInt
    enabled: bool

    @classmethod
    def _get_class(cls):
        return MonitorCallback


class MonitorCallback(Callback, HasConfig[MonitorCallbackConfig]):
    """
    Callback to monitor and log training performance (time and memory) by phase.

    Tracks multiple phases:
        - Total training
        - Per-batch loading
        - Forward and backward passes
        - Validation

    Metrics include time, CPU memory, and GPU memory usage.

    """

    _config_type = MonitorCallbackConfig

    def __init__(
        self,
        num_measurements: int = 100,
        warmup_iterations: int = 10,
        enabled: bool = True,
    ):
        self.config = self._config_type(
            num_measurements=num_measurements,
            warmup_iterations=warmup_iterations,
            enabled=enabled,
        )

        self.monitor_global_training = None
        self.monitor_epoch = None

        self.monitor_train_loop = None
        self.monitor_train_batch_loading = None
        self.monitor_forward = None
        self.monitor_backward = None
        self.monitor_optimization = None

        self.monitor_validation = None
        self.monitor_val_batch = None
        self.monitor_val_batch_loading = None
        self.monitor_evaluation = None

        self._optimization_config = None
        self._train_batch_size = None
        self._val_batch_size = None
        self._gpus_used = []
        self._n_iterations = 0

    @property
    def n_iterations(self) -> int:
        """
        Number of batches passed to the neural network.
        """
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, n: int):
        self._n_iterations = n
        if self.config.enabled:
            warm = self._n_iterations >= self.config.warmup_iterations
            self.monitor_train_loop.enabled = warm
            self.monitor_train_batch_loading.enabled = warm
            self.monitor_forward.enabled = warm
            self.monitor_backward.enabled = warm
            self.monitor_optimization.enabled = warm
            self.monitor_val_batch.enabled = warm
            self.monitor_val_batch_loading.enabled = warm
            self.monitor_evaluation.enabled = warm

    def on_exception(
        self,
        *,
        maps: Maps,
        state: TrainerState,
        exception: Exception,
        **kwargs,
    ) -> None:
        df = self._build_df()
        tsv_path = maps.training.splits[state.split_idx].logs.computational_tsv
        df.to_csv(tsv_path, sep=SEP)

        if isinstance(exception, torch.cuda.OutOfMemoryError) or OOM in str(exception):
            logger.error(
                "%s. To debug, you can have a look at your memory usage in %s",
                OOM,
                tsv_path,
            )

    def on_train_start(
        self,
        *,
        split: Split,
        optimization: OptimizationConfig,
        computational: ComputationalConfig,
        **kwargs,
    ) -> None:
        self._optimization_config = optimization
        self._train_batch_size = split.train_loader.batch_size
        self._val_batch_size = split.val_loader.batch_size

        if computational.gpu:
            self._gpus_used.append(torch.cuda.get_device_name(0))

        self.monitor_global_training = self._init_monitor(name=TRAIN, save_time=True)
        self.monitor_epoch = self._init_monitor(name=EPOCH)

        self.monitor_train_loop = self._init_monitor(
            gpu=computational.gpu, limited_measurements=True, name=TRAIN_LOOP
        )
        self.monitor_train_batch_loading = self._init_monitor(
            limited_measurements=True, name=TRAIN_LOAD
        )
        self.monitor_forward = self._init_monitor(
            gpu=computational.gpu,
            memory=True,
            limited_measurements=True,
            name=FORWARD,
        )
        self.monitor_backward = self._init_monitor(
            gpu=computational.gpu,
            memory=True,
            limited_measurements=True,
            name=BACKWARD,
        )
        self.monitor_optimization = self._init_monitor(
            gpu=computational.gpu,
            memory=True,
            limited_measurements=True,
            name=OPT,
        )

        self.monitor_validation = self._init_monitor(name=VAL)
        self.monitor_val_batch = self._init_monitor(
            gpu=computational.gpu, limited_measurements=True, name=VAL_LOOP
        )
        self.monitor_val_batch_loading = self._init_monitor(
            limited_measurements=True, name=VAL_LOAD
        )
        self.monitor_evaluation = self._init_monitor(
            gpu=computational.gpu, limited_measurements=True, name=EVAL
        )

        self.n_iterations = 0
        self.monitor_global_training.start()

    def on_epoch_start(self, **kwargs) -> None:
        self.monitor_epoch.start()
        if (
            self._optimization_config.accumulation_steps == 1
        ):  # otherwise, no optimization after the first batch
            self.monitor_train_loop.start()
        self.monitor_train_batch_loading.start()

    def on_forward_step_start(self, **kwargs) -> None:
        self.monitor_train_batch_loading.stop()
        self.monitor_forward.start()

    def on_backward_step_start(self, **kwargs) -> None:
        self.monitor_forward.stop()
        self.monitor_backward.start()

    def on_backward_step_end(self, **kwargs) -> None:
        self.monitor_backward.stop()
        self.n_iterations += 1

    def on_optimization_step_start(self, **kwargs) -> None:
        self.monitor_optimization.start()

    def on_optimization_step_end(self, **kwargs) -> None:
        self.monitor_optimization.stop()

    def on_batch_end(self, *, state: TrainerState, **kwargs) -> None:
        if state.stage == TrainerStage.TRAIN:
            if self.monitor_train_loop.running:
                self.monitor_train_loop.stop()
            if state.current_train_batch + 1 <= state.num_train_batches:
                if self._optimization_condition(state.current_train_batch + 1):
                    self.monitor_train_loop.start()
                self.monitor_train_batch_loading.start()
        elif state.called == TrainerCall.TRAIN and state.stage == TrainerStage.EVAL:
            self.monitor_val_batch.stop()
            if state.current_val_batch + 1 <= state.num_val_batches:
                self.monitor_val_batch.start()
                self.monitor_val_batch_loading.start()

    def on_validation_start(self, *, state: TrainerState, **kwargs) -> None:
        if state.called == TrainerCall.TRAIN:
            self.monitor_validation.start()
            self.monitor_val_batch.start()
            self.monitor_val_batch_loading.start()

    def on_evaluation_step_start(self, *, state: TrainerState, **kwargs) -> None:
        if state.called == TrainerCall.TRAIN:
            self.monitor_val_batch_loading.stop()
            self.monitor_evaluation.start()

    def on_evaluation_step_end(self, *, state: TrainerState, **kwargs) -> None:
        if state.called == TrainerCall.TRAIN:
            self.monitor_evaluation.stop()
            self.n_iterations += 1

    def on_validation_end(self, *, state: TrainerState, **kwargs) -> None:
        if state.called == TrainerCall.TRAIN:
            self.monitor_validation.stop()

    def on_epoch_end(self, **kwargs) -> None:
        self.monitor_epoch.stop()

    def on_train_end(self, *, maps: Maps, state: TrainerState, **kwargs) -> None:
        self.monitor_global_training.stop()

        df = self._build_df()
        maps.training.splits[state.split_idx].logs.create(exist_ok=True)
        df.to_csv(maps.training.splits[state.split_idx].logs.computational_tsv, sep=SEP)
        summary = TrainingSummary(maps.training.splits[state.split_idx].summary_log)
        summary.add_info(self._write_summary(df))

    def state_dict(self) -> Mapping[str, Any]:
        state_dict = {
            chrono.name: chrono.state_dict() for chrono in self._get_monitors()
        }
        state_dict[GPU] = self._gpus_used

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        for chrono in self._get_monitors():
            chrono.load_state_dict(state_dict[chrono.name])

        self._gpus_used = state_dict[GPU] + self._gpus_used

    def _get_monitors(self) -> Iterator[_PhaseMonitor]:
        """
        To iterate on all the monitors.
        """
        return (getattr(self, name) for name in dir(self) if name.startswith("monitor"))

    def _init_monitor(
        self,
        name: str,
        gpu: bool = False,
        memory: bool = False,
        limited_measurements: bool = False,
        save_time: bool = False,
    ) -> _PhaseMonitor:
        """
        Initializes a phase monitor.
        """
        return _PhaseMonitor(
            gpu=gpu,
            memory=memory,
            num_measurements=self.config.num_measurements
            if limited_measurements
            else None,
            enabled=self.config.enabled,
            save_time=save_time,
            name=name,
        )

    def _optimization_condition(self, batch_idx: int) -> bool:
        """
        If optimization will be performed on this batch.
        """
        return batch_idx % self._optimization_config.accumulation_steps == 0

    def _build_df(self) -> pd.DataFrame:
        """
        Gathers all the computational metrics in a DataFrame.
        """
        results = {
            _add_s_suffix(TRAIN): self.monitor_global_training.times,
            _add_s_suffix(EPOCH): self.monitor_epoch.times,
            _add_s_suffix(TRAIN_LOOP): self.monitor_train_loop.times,
            _add_s_suffix(TRAIN_LOAD): self.monitor_train_batch_loading.times,
            _add_s_suffix(FORWARD): self.monitor_forward.times,
            FORWARD_GPU: self.monitor_forward.gpu_times,
            FORWARD_MEM: np.array(self.monitor_forward.gpu_max_mem) / 1024**2,
            _add_s_suffix(BACKWARD): self.monitor_backward.times,
            BACKWARD_GPU: self.monitor_backward.gpu_times,
            BACKWARD_MEM: np.array(self.monitor_backward.gpu_max_mem) / 1024**2,
            _add_s_suffix(OPT): self.monitor_optimization.times,
            OPT_GPU: self.monitor_optimization.gpu_times,
            OPT_MEM: np.array(self.monitor_optimization.gpu_max_mem) / 1024**2,
            _add_s_suffix(VAL): self.monitor_validation.times,
            _add_s_suffix(VAL_LOOP): self.monitor_val_batch.times,
            _add_s_suffix(VAL_LOAD): self.monitor_val_batch.times,
            _add_s_suffix(EVAL): self.monitor_evaluation.times,
            EVAL_GPU: self.monitor_evaluation.gpu_times,
            EVAL_MEM: np.array(self.monitor_evaluation.gpu_max_mem) / 1024**2,
        }
        df = pd.DataFrame({name: pd.Series(col) for name, col in results.items()})
        df = df.rename_axis(index="measurement #")

        return df

    def _write_summary(self, df: pd.DataFrame) -> str:
        """
        Write the computational metrics to a summary file from the DataFrame returned
        by :py:meth:`_build_df`.
        """

        lines = [
            "\n*********************** Computational Summary (mean ± std [n measurements]) ***********************\n"
        ]
        if self._gpus_used:
            lines.append(
                f"""GPU: {" then ".join([f"'{gpu}'" for gpu in self._gpus_used])}\n"""
            )
        lines.append(
            f"{'Phase':^15} | {'Time (s)':^25} | {'GPU Time (s)':^25} | {'GPU Max Memory (MB)':^25}\n"
            + "-" * 99
        )

        lines.append(
            f"{'Training':<15} | {_repr_series(df[_add_s_suffix(TRAIN)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' + 'Epoch':<15} | {_repr_series(df[_add_s_suffix(EPOCH)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' * 2 + 'Iteration':<15} | {_repr_series(df[_add_s_suffix(TRAIN_LOOP)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' * 3 + 'Data loading':<15} | {_repr_series(df[_add_s_suffix(TRAIN_LOAD)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' * 3 + 'Forward':<15} | {_repr_series(df[_add_s_suffix(FORWARD)]):<25} | {_repr_series(df[FORWARD_GPU]):<25} | {_repr_series(df[FORWARD_MEM]):<25}"
        )
        lines.append(
            f"{' ' * 3 + 'Backward':<15} | {_repr_series(df[_add_s_suffix(BACKWARD)]):<25} | {_repr_series(df[BACKWARD_GPU]):<25} | {_repr_series(df[BACKWARD_MEM]):<25}"
        )
        lines.append(
            f"{' ' * 3 + 'Optimization':<15} | {_repr_series(df[_add_s_suffix(OPT)]):<25} | {_repr_series(df[OPT_GPU]):<25} | {_repr_series(df[OPT_MEM]):<25}"
        )

        lines.append(
            f"\nThroughput: {self._train_batch_size / df[_add_s_suffix(TRAIN_LOOP)].mean():.2f} images/s"
        )
        if self._gpus_used:
            total_gpu_time = (
                df[FORWARD_GPU].mean() + df[BACKWARD_GPU].mean() + df[OPT_GPU].mean()
            )
            lines.append(
                f"GPU throughput: {self._train_batch_size / total_gpu_time:.2f} images/s"
            )

        lines.append("-" * 99)
        lines.append(
            f"{'Validation':<15} | {_repr_series(df[_add_s_suffix(VAL)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' + 'Iteration':<15} | {_repr_series(df[_add_s_suffix(VAL_LOOP)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' * 2 + 'Data loading':<15} | {_repr_series(df[_add_s_suffix(VAL_LOAD)]):<25} | {'':<25} | {'':<25}"
        )
        lines.append(
            f"{' ' * 2 + 'Evaluation':<15} | {_repr_series(df[_add_s_suffix(EVAL)]):<25} | {_repr_series(df[EVAL_GPU]):<25} | {_repr_series(df[EVAL_MEM]):<25}"
        )

        lines.append(
            f"\nThroughput: {self._val_batch_size / df[_add_s_suffix(VAL_LOOP)].mean():.2f} images/s"
        )
        if self._gpus_used:
            lines.append(
                f"GPU throughput: {self._val_batch_size / df[EVAL_GPU].mean():.2f} images/s"
            )

        lines.append("*" * 99 + "\n")

        return "\n".join(lines)


def _repr_series(samples: pd.Series) -> str:
    """
    To represent a series of measurements.
    """
    samples = samples.dropna()
    if len(samples) == 0:
        return ""
    elif len(samples) > 1:
        return f"{samples.mean():.2e} ± {samples.std():.2e} [{len(samples)}]"
    return f"{samples.mean():.2e} [1]"


def _add_s_suffix(input_str: str) -> str:
    """
    Adds ' (s)' to an input string.
    """
    return input_str + " (s)"


class _PhaseMonitor:
    to_save = ["times", "gpu_times", "gpu_max_mem", "n_measured"]
    """
    To monitor a training phase:
        - execution time;
        - execution time of GPU tasks;
        - GPU peak memory usage.

    The class allows repetitions of measurements.
    """

    def __init__(
        self,
        gpu: bool,
        memory: bool = True,
        num_measurements: Optional[int] = None,
        save_time: bool = False,
        enabled: bool = True,
        name: Optional[str] = None,
    ):
        self.gpu = gpu
        self.memory = memory
        self.num_measurements = num_measurements
        self.save_time = save_time
        self.enabled = enabled
        self.name = name

        self.times = []
        self.gpu_times = []
        self.gpu_max_mem = []
        self.n_measured = 0
        self._start_time = None
        self._gpu_start_event = None
        self._gpu_end_event = None

        self.running = False

    def start(self):
        """
        Starts the phase.
        """
        if not self.enabled or (
            self.num_measurements and self.n_measured >= self.num_measurements
        ):
            return

        if self.gpu:
            self._gpu_start_event = torch.cuda.Event(enable_timing=True)
            self._gpu_end_event = torch.cuda.Event(enable_timing=True)

            if self.memory:
                torch.cuda.reset_peak_memory_stats()

            self._gpu_start_event.record()

        self._start_time = time.perf_counter()

        self.running = True

    def stop(self):
        """
        Ends the phase.
        """
        if (
            not self.enabled
            or (self.num_measurements and self.n_measured >= self.num_measurements)
            or not self.running
        ):
            return

        if self.gpu:
            self._gpu_end_event.record()
            torch.cuda.synchronize()
            self.gpu_times.append(
                self._gpu_start_event.elapsed_time(self._gpu_end_event)
            )

        elapsed = time.perf_counter() - self._start_time
        self.times.append(elapsed)

        if self.gpu and self.memory:
            self.gpu_max_mem.append(torch.cuda.max_memory_allocated())

        self.n_measured += 1
        self.running = False

    def state_dict(self) -> Mapping[str, Any]:
        """
        Returns the current state of the monitoring.
        """
        state_dict = {}
        if self.save_time and self.running:
            state_dict["elapsed"] = time.perf_counter() - self._start_time
        for name in self.to_save:
            state_dict[name] = getattr(self, name)

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """
        Reinitialize the monitoring to a certain state.
        """
        for name in self.to_save:
            setattr(self, name, state_dict[name])
        if self.save_time and self.running:
            self._start_time -= state_dict["elapsed"]
