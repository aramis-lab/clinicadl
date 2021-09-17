# TODO: Integrate printed logs ?
import logging
from os import makedirs, path

import numpy as np
import pandas as pd


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def setup_logging(verbosity: int = 0) -> None:
    """
    Setup ClinicaDL's logging facilities.
    Args:
        verbosity: The desired level of verbosity for logging.
            (0 (default): WARNING, 1: INFO, 2: DEBUG)
    """
    from logging import DEBUG, INFO, WARNING, Formatter, StreamHandler, getLogger
    from sys import stderr, stdout

    # Cap max verbosity level to 2.
    verbosity = min(verbosity, 2)

    # Define the module level logger.
    logger = getLogger("clinicadl")
    logger.setLevel([WARNING, INFO, DEBUG][verbosity])

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout = StreamHandler(stdout)
    stdout.addFilter(StdLevelFilter())
    stderr = StreamHandler(stderr)
    stderr.addFilter(StdLevelFilter(err=True))
    # create formatter
    formatter = Formatter("%(asctime)s - %(levelname)s: %(message)s", "%H:%M:%S")
    # add formatter to ch
    stdout.setFormatter(formatter)
    stderr.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(stdout)
    logger.addHandler(stderr)
    logger.propagate = False


class LogWriter:
    def __init__(
        self,
        maps_path,
        evaluation_metrics,
        fold,
        resume=False,
        beginning_epoch=0,
        network=None,
    ):
        from time import time

        from torch.utils.tensorboard import SummaryWriter

        # Generate columns of DataFrame
        columns_train = [
            selection.split("-")[0] + "_train" for selection in evaluation_metrics
        ]
        columns_valid = [
            selection.split("-")[0] + "_valid" for selection in evaluation_metrics
        ]
        self.columns = ["epoch", "iteration", "time"] + columns_train + columns_valid

        self.evaluation_metrics = evaluation_metrics
        self.maps_path = maps_path

        self.file_dir = path.join(self.maps_path, f"fold-{fold}", "training_logs")
        if network is not None:
            self.file_dir = path.join(self.file_dir, f"network-{network}")
        makedirs(self.file_dir, exist_ok=True)
        tsv_path = path.join(self.file_dir, "training.tsv")

        self.beginning_epoch = beginning_epoch
        if not resume:
            results_df = pd.DataFrame(columns=self.columns)
            with open(tsv_path, "w") as f:
                results_df.to_csv(f, index=False, sep="\t")
            self.beginning_time = time()
        else:
            if not path.exists(tsv_path):
                raise ValueError(
                    f"The training.tsv file of the fold {fold} in the MAPS "
                    f"{self.maps_path} does not exist."
                )
            truncated_tsv = pd.read_csv(tsv_path, sep="\t")
            truncated_tsv.set_index(["epoch", "iteration"], inplace=True)
            truncated_tsv.drop(self.beginning_epoch, level=0, inplace=True)
            self.beginning_time = time() + truncated_tsv.iloc[-1, -1]
            truncated_tsv.to_csv(tsv_path, index=True, sep="\t")

        self.writer_train = SummaryWriter(
            path.join(self.file_dir, "tensorboard", "train")
        )
        self.writer_valid = SummaryWriter(
            path.join(self.file_dir, "tensorboard", "validation")
        )

    def step(self, epoch, i, metrics_train, metrics_valid, len_epoch):
        """
        Write a new row on the output file training.tsv.

        Args:
            epoch (int): current epoch number
            i (int): current iteration number
            metrics_train (Dict[str:float]): metrics on the training set
            metrics_valid (Dict[str:float]): metrics on the validation set
            len_epoch (int): number of iterations in an epoch
        """
        from time import time

        # Write TSV file
        tsv_path = path.join(self.file_dir, "training.tsv")

        t_current = time() - self.beginning_time
        general_row = [epoch, i, t_current]
        train_row = list()
        valid_row = list()
        for selection in self.evaluation_metrics:
            if selection in metrics_train:
                train_row.append(metrics_train[selection])
                valid_row.append(metrics_valid[selection])
            else:
                # Multi-class case, there is one metric per class (i.e. sensitivity-0, sensitivity-1...)
                train_values = [
                    metrics_train[key]
                    for key in metrics_train.keys()
                    if selection in key
                ]
                valid_values = [
                    metrics_valid[key]
                    for key in metrics_valid.keys()
                    if selection in key
                ]
                train_row.append(np.mean(train_values))
                valid_row.append(np.mean(valid_values))

        row = [general_row + train_row + valid_row]
        row_df = pd.DataFrame(row, columns=self.columns)
        with open(tsv_path, "a") as f:
            row_df.to_csv(f, header=False, index=False, sep="\t")

        # Write tensorboard logs
        global_step = i + epoch * len_epoch
        for metric_idx, metric in enumerate(self.evaluation_metrics):
            self.writer_train.add_scalar(
                metric,
                train_row[metric_idx],
                global_step,
            )
            self.writer_valid.add_scalar(
                metric,
                valid_row[metric_idx],
                global_step,
            )
