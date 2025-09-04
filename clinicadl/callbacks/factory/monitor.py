import os
import statistics
import time
from pathlib import Path

import psutil
import torch

from clinicadl.callbacks.training_state import _TrainingState

from .base import Callback


class PhaseMonitor:
    """
    Monitor for tracking time and memory usage of a specific training phase.

    This class records:
        - Execution time per iteration (with mean, min, std)
        - CPU memory usage (resident set size)
        - GPU memory usage (if CUDA is available)

    Methods
    -------
    start()
        Starts the timing of the phase.
    stop()
        Stops the timing and logs memory usage.
    summary()
        Returns statistics (min, mean, std) for time and memory.
    """

    def __init__(self):
        self.times = []
        self.cpu_mem = []
        self.gpu_mem = []
        self.process = psutil.Process(os.getpid())
        self._start_time = None

    def start(self):
        """Start the timer for the phase."""
        self._start_time = time.perf_counter()

    def stop(self):
        """
        Stop the timer and log memory usage.

        Raises
        ------
        RuntimeError
            If stop is called without a corresponding start.
        """
        if self._start_time is None:
            raise RuntimeError("You must call start() before stop().")
        elapsed = time.perf_counter() - self._start_time
        self.times.append(elapsed)

        # Memory recording
        self.cpu_mem.append(self.process.memory_info().rss / 1e6)  # in MB
        if torch.cuda.is_available():
            self.gpu_mem.append(torch.cuda.memory_allocated() / 1e6)  # in MB

        self._start_time = None

    def summary(self):
        """
        Compute min, average and standard deviation for time, CPU, and GPU memory.

        Returns
        -------
        dict
            A dictionary with stats for 'time', 'cpu', and 'gpu' usage.
        """

        def stats(lst):
            if not lst:
                return 0.0, 0.0, 0.0
            return (
                min(lst),
                statistics.mean(lst),
                statistics.stdev(lst) if len(lst) > 1 else 0.0,
            )

        t_min, t_avg, t_std = stats(self.times)
        c_min, c_avg, c_std = stats(self.cpu_mem)
        g_min, g_avg, g_std = stats(self.gpu_mem)

        return {
            "time": {"min": t_min, "avg": t_avg, "std": t_std},
            "cpu": {"min": c_min, "avg": c_avg, "std": c_std},
            "gpu": {"min": g_min, "avg": g_avg, "std": g_std},
        }


class _Monitor(Callback):
    """
    Callback to monitor and log training performance (time and memory) by phase.

    Tracks multiple phases:
        - Total training
        - Per-batch loading
        - Forward and backward passes
        - Validation

    Metrics include time, CPU memory, and GPU memory usage.

    """

    def __init__(self):
        self.all_phases = PhaseMonitor()
        self.training_phase = PhaseMonitor()
        self.validation_phase = PhaseMonitor()
        self.loading_phase = PhaseMonitor()
        self.forward_phase = PhaseMonitor()
        self.backward_phase = PhaseMonitor()

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        self.all_phases.start()

    def on_epoch_begin(self, config: _TrainingState, **kwargs) -> None:
        self.loading_phase.start()

    def on_batch_begin(self, config: _TrainingState, **kwargs) -> None:
        self.loading_phase.stop()

        self.training_phase.start()
        self.forward_phase.start()

    def on_backward_begin(self, config: _TrainingState, **kwargs) -> None:
        self.forward_phase.stop()
        self.backward_phase.start()

    def on_backward_end(self, config: _TrainingState, **kwargs) -> None:
        self.backward_phase.stop()

    def on_batch_end(self, config: _TrainingState, **kwargs) -> None:
        self.training_phase.stop()
        self.loading_phase.start()

    def on_validation_begin(self, config: _TrainingState, **kwargs) -> None:
        self.validation_phase.start()

    def on_validation_end(self, config: _TrainingState, **kwargs) -> None:
        self.validation_phase.stop()

    def on_train_end(self, config: _TrainingState, **kwargs) -> None:
        self.all_phases.stop()
        self.write_file(config.maps.training.splits[config.split.index].performance_txt)

    def write_file(self, file_txt: Path):
        """
        Write the performance metrics to a text file.

        Parameters
        ----------
        file_txt : Path
            File path where the performance report will be saved.
        """

        monitor = {
            "all": self.all_phases,
            "training": self.training_phase,
            "validation": self.validation_phase,
            "loading": self.loading_phase,
            "forward": self.forward_phase,
            "backward": self.backward_phase,
        }
        header = (
            "\n=== Performance Summary ===\n\n"
            f"{'Phase':<12} | {'Time (s)':<25} | {'CPU Memory (MB)':<25} | {'GPU Memory (MB)':<25}\n"
            + "-"
            * 96
        )
        lines = [header]
        for phase, m in monitor.items():
            s = m.summary()

            # Handle GPU display
            if torch.cuda.is_available():
                gpu_str = f"{s['gpu']['avg']:.1f} ± {s['gpu']['std']:.1f} (min {s['gpu']['min']:.1f})"
            else:
                gpu_str = "N/A"

            line = (
                f"{phase.capitalize():<12} | "
                f"{s['time']['avg']:.2f} ± {s['time']['std']:.2f} (min {s['time']['min']:.2f})".ljust(
                    25
                )
                + " | "
                f"{s['cpu']['avg']:.1f} ± {s['cpu']['std']:.1f} (min {s['cpu']['min']:.1f})".ljust(
                    25
                )
                + " | "
                f"{gpu_str}"
            )
            lines.append(line)

        lines.append("=" * 96 + "\n")

        with open(file_txt, "w") as f:
            f.write("\n".join(lines))
