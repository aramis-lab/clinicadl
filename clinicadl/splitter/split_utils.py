from pathlib import Path
from typing import List

from clinicadl.utils.exceptions import ClinicaDLArgumentError


def find_splits(maps_path: Path, split_name: str) -> List[int]:
    """Find which splits that were trained in the MAPS."""
    splits = [
        int(split.name.split("-")[1])
        for split in list(maps_path.iterdir())
        if split.name.startswith(f"{split_name}-")
    ]
    return splits


def find_stopped_splits(maps_path: Path, split_name: str) -> List[int]:
    """Find which splits for which training was not completed."""
    existing_split_list = find_splits(maps_path, split_name)
    stopped_splits = [
        split
        for split in existing_split_list
        if (maps_path / f"{split_name}-{split}" / "tmp")
        in list((maps_path / f"{split_name}-{split}").iterdir())
    ]
    return stopped_splits


def find_finished_splits(maps_path: Path, split_name: str) -> List[int]:
    """Find which splits for which training was completed."""
    finished_splits = list()
    existing_split_list = find_splits(maps_path, split_name)
    stopped_splits = find_stopped_splits(maps_path, split_name)
    for split in existing_split_list:
        if split not in stopped_splits:
            performance_dir_list = [
                performance_dir
                for performance_dir in list(
                    (maps_path / f"{split_name}-{split}").iterdir()
                )
                if "best-" in performance_dir.name
            ]
            if len(performance_dir_list) > 0:
                finished_splits.append(split)
    return finished_splits


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
