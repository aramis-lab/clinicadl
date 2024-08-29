from pathlib import Path
from typing import List, Optional

import pandas as pd

from clinicadl.splitter.split_utils import print_description_log
from clinicadl.utils.exceptions import ClinicaDLArgumentError, MAPSError


def find_selection_metrics(maps_path: Path, split_name: str, split):
    """Find which selection metrics are available in MAPS for a given split."""

    split_path = maps_path / f"{split_name}-{split}"
    if not split_path.is_dir():
        raise KeyError(
            f"Training of split {split} was not performed."
            f"Please execute maps_manager.train(split_list=[{split}])"
        )

    return [
        metric.name.split("-")[1]
        for metric in list(split_path.iterdir())
        if metric.name.startswith("best-")
    ]


def check_selection_metric(
    maps_path: Path, split_name: str, split, selection_metric=None
):
    """Check that a given selection metric is available for a given split."""
    available_metrics = find_selection_metrics(maps_path, split_name, split)

    if not selection_metric:
        if len(available_metrics) > 1:
            raise ClinicaDLArgumentError(
                f"Several metrics are available for split {split}. "
                f"Please choose which one you want to read among {available_metrics}"
            )
        else:
            selection_metric = available_metrics[0]
    else:
        if selection_metric not in available_metrics:
            raise ClinicaDLArgumentError(
                f"The metric {selection_metric} is not available."
                f"Please choose among is the available metrics {available_metrics}."
            )
    return selection_metric


def get_metrics(
    maps_path: Path,
    split_name: str,
    data_group: str,
    split: int = 0,
    selection_metric: Optional[str] = None,
    mode: str = "image",
    verbose: bool = True,
):
    """
    Get the metrics corresponding to a group of participants identified by its data_group.

    Args:
        data_group (str): name of the data group used for the prediction task.
        split (int): Index of the split used for training.
        selection_metric (str): Metric used for best weights selection.
        mode (str): level of the prediction
        verbose (bool): if True will print associated prediction.log
    Returns:
        (dict[str:float]): Values of the metrics
    """
    selection_metric = check_selection_metric(
        maps_path, split_name, split, selection_metric
    )
    if verbose:
        print_description_log(
            maps_path, split_name, data_group, split, selection_metric
        )
    prediction_dir = (
        maps_path / f"{split_name}-{split}" / f"best-{selection_metric}" / data_group
    )
    if not prediction_dir.is_dir():
        raise MAPSError(
            f"No prediction corresponding to data group {data_group} was found."
        )
    df = pd.read_csv(
        prediction_dir / f"{data_group}_{mode}_level_metrics.tsv", sep="\t"
    )
    return df.to_dict("records")[0]
