from pathlib import Path

import numpy as np
import pandas as pd


class LogWriter:
    """
    Write training logs in the MAPS
    """

    def __init__(
        self,
        maps_path: Path,
        evaluation_metrics,
        split,
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

        self.file_dir = self.maps_path / f"split-{split}" / "training_logs"
        if network is not None:
            self.file_dir = self.file_dir / f"network-{network}"
        self.file_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = self.file_dir / "training.tsv"

        self.beginning_epoch = beginning_epoch
        if not resume:
            results_df = pd.DataFrame(columns=self.columns)
            with tsv_path.open(mode="w") as f:
                results_df.to_csv(f, index=False, sep="\t")
            self.beginning_time = time()
        else:
            if not tsv_path.is_file():
                raise FileNotFoundError(
                    f"The training.tsv file of the split {split} in the MAPS "
                    f"{self.maps_path} does not exist."
                )
            truncated_tsv = pd.read_csv(tsv_path, sep="\t")
            truncated_tsv.set_index(["epoch", "iteration"], inplace=True)
            truncated_tsv.drop(self.beginning_epoch, level=0, inplace=True)
            if len(truncated_tsv) == 0:
                self.beginning_time = 0
            else:
                self.beginning_time = time() + truncated_tsv.iloc[-1, 0]
            truncated_tsv.to_csv(tsv_path, index=True, sep="\t")

        self.writer_train = SummaryWriter(self.file_dir / "tensorboard" / "train")
        self.writer_valid = SummaryWriter(self.file_dir / "tensorboard" / "validation")

    def step(self, epoch, i, metrics_train, metrics_valid, len_epoch, file_name=None):
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

        if file_name:
            tsv_path = self.file_dir / file_name
        else:
            tsv_path = self.file_dir / "training.tsv"

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
        with tsv_path.open(mode="a") as f:
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
