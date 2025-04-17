import json
from datetime import datetime, timedelta
from time import time
from typing import List, Optional

import numpy as np


class Chronometer:
    """
    A lightweight profiler for timing a PyTorch training loop.

    Tracks durations of training phases like data loading, forward pass,
    backward pass, and validation. Supports summary display and export.

    Example
    -------
    chrono = Chronometer()
    chrono.start()

    for epoch in range(epochs):
        chrono.next_iter()
        for i, (x, y) in enumerate(train_loader):
            chrono.forward()
            ...
            chrono.backward()
            loss.backward()
            optimizer.step()
            chrono.update()

        chrono.validation()
        for val_x, val_y in val_loader:
            ...
        chrono.validation()

        chrono.next_iter()

    chrono.stop()
    chrono.display()
    """

    def __init__(self) -> None:
        self.time_perf_train: List[float] = []
        self.time_perf_load: List[float] = []
        self.time_perf_forward: List[float] = []
        self.time_perf_backward: List[float] = []
        self.power: List[float] = []

        self.start_proc: Optional[datetime] = None
        self.stop_proc: Optional[datetime] = None

        self.start_training: Optional[float] = None
        self.start_dataload: Optional[float] = None
        self.start_backward: Optional[float] = None
        self.start_forward: Optional[float] = None
        self.start_valid: Optional[datetime] = None

        self.val_time: Optional[timedelta] = None
        self.time_point: Optional[float] = None

    def tac_time(self, clear: bool = False) -> Optional[float]:
        """
        Measures time elapsed since last call.

        Parameters
        ----------
        clear : bool
            If True, resets the reference point.

        Returns
        -------
        float or None
            Elapsed time in seconds, or None if cleared.
        """
        if self.time_point is None or clear:
            self.time_point = time()
            return None
        else:
            new_time = time() - self.time_point
            self.time_point = time()
            return new_time

    def clear(self) -> None:
        """Clears all recorded timing data."""
        self.time_perf_train.clear()
        self.time_perf_load.clear()
        self.time_perf_forward.clear()
        self.time_perf_backward.clear()

    def start(self) -> None:
        """Marks the beginning of the overall training."""
        self.start_proc = datetime.now()

    def stop(self) -> None:
        """Marks the end of the overall training."""
        self.stop_proc = datetime.now()

    def elapsed(self) -> Optional[float]:
        """
        Returns time elapsed since `start()` was called.

        Returns
        -------
        float or None
            Elapsed time in seconds, or None if not started.
        """
        if self.start_proc is None:
            return None
        return (datetime.now() - self.start_proc).total_seconds()

    def _dataload(self) -> None:
        if self.start_dataload is None:
            self.start_dataload = time()
        else:
            self.time_perf_load.append(time() - self.start_dataload)
            self.start_dataload = None

    def _training(self) -> None:
        if self.start_training is None:
            self.start_training = time()
        else:
            self.time_perf_train.append(time() - self.start_training)
            self.start_training = None

    def _forward(self) -> None:
        if self.start_forward is None:
            self.start_forward = time()
        else:
            self.time_perf_forward.append(time() - self.start_forward)
            self.start_forward = None

    def _backward(self) -> None:
        if self.start_backward is None:
            self.start_backward = time()
        else:
            self.time_perf_backward.append(time() - self.start_backward)
            self.start_backward = None

    def next_iter(self) -> None:
        """Call this at the end of an iteration to finalize dataload timing."""
        self._dataload()

    def forward(self) -> None:
        """
        Call this before and after the forward pass.
        Handles dataloading, training, and forward time tracking.
        """
        self._dataload()
        self._training()
        self._forward()

    def backward(self) -> None:
        """
        Call this before and after the backward pass.
        Handles forward and backward time tracking.
        """
        self._forward()
        self._backward()

    def update(self) -> None:
        """
        Call this after the optimizer step.
        Ends backward and training timing.
        """
        self._backward()
        self._training()

    def validation(self) -> None:
        """
        Call this before and after the validation phase.
        Measures total validation duration.
        """
        if self.start_valid is None:
            self.start_valid = datetime.now()
        else:
            self.val_time = datetime.now() - self.start_valid
            self.start_valid = None

    def display(self) -> None:
        """
        Displays collected timing statistics and performance summary.
        """
        if self.stop_proc and self.start_proc:
            print(">>> Training complete in:", str(self.stop_proc - self.start_proc))

        if self.time_perf_train:
            print(
                ">>> Training performance time: min {:.4f}, avg {:.4f} (+/- {:.4f})".format(
                    np.min(self.time_perf_train[1:]),
                    np.median(self.time_perf_train[1:]),
                    np.std(self.time_perf_train[1:]),
                )
            )

        if self.time_perf_load:
            print(
                ">>> Loading performance time: min {:.4f}, avg {:.4f} (+/- {:.4f})".format(
                    np.min(self.time_perf_load[1:]),
                    np.mean(self.time_perf_load[1:]),
                    np.std(self.time_perf_load[1:]),
                )
            )

        if self.time_perf_forward:
            print(
                ">>> Forward performance time: avg {:.4f} (+/- {:.4f})".format(
                    np.mean(self.time_perf_forward[1:]),
                    np.std(self.time_perf_forward[1:]),
                )
            )

        if self.time_perf_backward:
            print(
                ">>> Backward performance time: avg {:.4f} (+/- {:.4f})".format(
                    np.mean(self.time_perf_backward[1:]),
                    np.std(self.time_perf_backward[1:]),
                )
            )

        if self.power:
            print(">>> Peak Power during training: {:.2f} W".format(np.max(self.power)))

        if self.val_time:
            print(">>> Validation time:", self.val_time)

        if self.time_perf_train and self.time_perf_load:
            print(">>> Sortie trace #####################################")
            print(
                ">>> JSON",
                json.dumps(
                    {
                        "GPU process - Forward/Backward": self.time_perf_train,
                        "CPU process - Dataloader": self.time_perf_load,
                    }
                ),
            )
