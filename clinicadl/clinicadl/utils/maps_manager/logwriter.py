# TODO: Integrate printed logs ?
import logging
from os import makedirs, path

import pandas as pd


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


class LogWriter:
    def __init__(self, maps_path, evaluation_metrics):

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

    def init_fold(self, fold, resume=False, beginning_epoch=0):
        from time import time

        from torch.utils.tensorboard import SummaryWriter

        file_dir = path.join(self.maps_path, f"fold-{fold}", "training_logs")
        makedirs(file_dir, exist_ok=True)
        tsv_path = path.join(file_dir, "training.tsv")

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
            self.beginning_time = time() + training_tsv.iloc[-1, -1]
            truncated_tsv.to_csv(tsv_path, index=True, sep="\t")

        self.writer_train = SummaryWriter(path.join(file_dir, "tensorboard", "train"))
        self.writer_valid = SummaryWriter(
            path.join(file_dir, "tensorboard", "validation")
        )

    def step(self, fold, epoch, i, metrics_train, metrics_valid, len_epoch):
        """
        Write a new row on the output file training.tsv.

        Args:
            fold (int): number of fold
            epoch (int): current epoch number
            i (int): current iteration number
            metrics_train (Dict[str:float]): metrics on the training set
            metrics_valid (Dict[str:float]): metrics on the validation set
            len_epoch (int): number of iterations in an epoch
        """
        from time import time

        file_dir = path.join(self.maps_path, f"fold-{fold}", "training_logs")

        # Write TSV file
        tsv_path = path.join(file_dir, "training.tsv")

        t_current = time() - self.beginning_time
        general_row = [epoch, i, t_current]
        train_row = [metrics_train[selection] for selection in self.evaluation_metrics]
        valid_row = [metrics_valid[selection] for selection in self.evaluation_metrics]

        row = [general_row + train_row + valid_row]
        row_df = pd.DataFrame(row, columns=self.columns)
        with open(tsv_path, "a") as f:
            row_df.to_csv(f, header=False, index=False, sep="\t")

        # Write tensorboard logs
        global_step = i + epoch * len_epoch
        for selection in self.evaluation_metrics:
            self.writer_train.add_scalar(
                selection,
                metrics_train[selection],
                global_step,
            )
            self.writer_valid.add_scalar(
                selection,
                metrics_valid[selection],
                global_step,
            )
