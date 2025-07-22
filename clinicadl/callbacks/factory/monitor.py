import os
import statistics
import time

import psutil
import torch

from clinicadl.callbacks.training_state import _TrainingState

from ..base import Callback


class PhaseMonitor:
    def __init__(self):
        self.times = []
        self.cpu_mem = []
        self.gpu_mem = []
        self.process = psutil.Process(os.getpid())
        self._start_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
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
    def __init__(self):
        self.training_phase = PhaseMonitor()

        self.validation_phase = PhaseMonitor()
        self.loading_phase = PhaseMonitor()
        self.forward_phase = PhaseMonitor()
        self.backward_phase = PhaseMonitor()

    def on_train_begin(self, config: _TrainingState, **kwargs) -> None:
        self.training_phase.start()
        self.validation_phase.start()
        self.loading_phase.start()
        self.forward_phase.start()
        self.backward_phase.start()
