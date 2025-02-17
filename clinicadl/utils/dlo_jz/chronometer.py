import json
from datetime import datetime
from time import time

import numpy as np
from pynvml.smi import nvidia_smi

###############################
# Author : Bertrand CABOT from IDRIS(CNRS)
#
########################


class Chronometer:
    """
    A light profiler to time a pytorch training loop

    Methods
    -------
    power_measurement(self)
        get the power measurement at time point from CUDA nvsmi
    tac_time(self, clear=False)
        like a stopwatch, get time difference between each call
    clear(self)
        clear all timer
    display(self)
        print all the traces (in out log)
    ...

    Example
    -------
    chrono = Chronometer()

    chrono.start()
    ...
    for epoch in range(args.epochs):

        chrono.next_iter()

        for i, (samples, labels) in enumerate(train_loader):

            chrono.forward()

            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(samples, labels)

            chrono.backward()

            loss.backward()
            optimizer.step()

            chrono.update()
            ...
            #### VALIDATION ############
            chrono.validation()
            for iv, (val_images, val_labels) in enumerate(val_loader):
                ....
            chrono.validation()
            #### END OF VALIDATION ############

            chrono.next_iter()

    chrono.stop()
    """

    def __init__(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []
        self.power = []
        self.start_proc = None
        self.stop_proc = None
        self.start_training = None
        self.start_dataload = None
        self.start_backward = None
        self.start_forward = None
        self.start_valid = None
        self.val_time = None
        self.time_point = None
        self.nvsmi = nvidia_smi.getInstance()

    def power_measurement(self):
        powerquery = self.nvsmi.DeviceQuery("power.draw")["gpu"]
        for g in range(len(powerquery)):
            self.power.append(powerquery[g]["power_readings"]["power_draw"])

    def tac_time(self, clear=False):
        if self.time_point is None or clear:
            self.time_point = time()
            return
        else:
            new_time = time() - self.time_point
            self.time_point = time()
            return new_time

    def clear(self):
        self.time_perf_train = []
        self.time_perf_load = []
        self.time_perf_forward = []
        self.time_perf_backward = []

    def start(self):
        self.start_proc = datetime.now()

    def stop(self):
        self.stop_proc = datetime.now()

    def _dataload(self):
        if self.start_dataload is None:
            self.start_dataload = time()
        else:
            self.time_perf_load.append(time() - self.start_dataload)
            self.start_dataload = None

    def _training(self):
        if self.start_training is None:
            self.start_training = time()
        else:
            self.time_perf_train.append(time() - self.start_training)
            self.start_training = None

    def _forward(self):
        if self.start_forward is None:
            self.start_forward = time()
        else:
            self.time_perf_forward.append(time() - self.start_forward)
            self.start_forward = None

    def _backward(self):
        if self.start_backward is None:
            self.start_backward = time()
        else:
            self.time_perf_backward.append(time() - self.start_backward)
            self.start_backward = None

    def next_iter(self):
        self._dataload()

    def forward(self):
        self._dataload()
        self._training()
        self._forward()

    def backward(self):
        self._forward()
        self._backward()

    def update(self):
        self._backward()
        self.power_measurement()
        self._training()

    def validation(self):
        if self.start_valid is None:
            self.start_valid = datetime.now()
        else:
            self.val_time = datetime.now() - self.start_valid
            self.start_valid = None

    def display(self):
        if self.stop_proc and self.start_proc:
            print(">>> Training complete in: " + str(self.stop_proc - self.start_proc))
        if len(self.time_perf_train) > 0:
            print(
                ">>> Training performance time: min {} avg {} seconds (+/- {})".format(
                    np.min(self.time_perf_train[1:]),
                    np.median(self.time_perf_train[1:]),
                    np.std(self.time_perf_train[1:]),
                )
            )
        if len(self.time_perf_load) > 0:
            print(
                ">>> Loading performance time: min {} avg {} seconds (+/- {})".format(
                    np.min(self.time_perf_load[1:]),
                    np.mean(self.time_perf_load[1:]),
                    np.std(self.time_perf_load[1:]),
                )
            )
        if len(self.time_perf_forward) > 0:
            print(
                ">>> Forward performance time: {} seconds (+/- {})".format(
                    np.mean(self.time_perf_forward[1:]),
                    np.std(self.time_perf_forward[1:]),
                )
            )
        if len(self.time_perf_backward) > 0:
            print(
                ">>> Backward performance time: {} seconds (+/- {})".format(
                    np.mean(self.time_perf_backward[1:]),
                    np.std(self.time_perf_backward[1:]),
                )
            )
        if len(self.power) > 0:
            print(">>> Peak Power during training: {} W)".format(np.max(self.power)))
        if self.val_time:
            print(">>> Validation time: {}".format(self.val_time))
        if len(self.time_perf_train) > 0 and len(self.time_perf_load) > 0:
            print(">>> Sortie trace #####################################")
            print(
                ">>>JSON",
                json.dumps(
                    {
                        "GPU process - Forward/Backward": self.time_perf_train,
                        "CPU process - Dataloader": self.time_perf_load,
                    }
                ),
            )
