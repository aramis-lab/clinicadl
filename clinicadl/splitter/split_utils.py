from pathlib import Path
from typing import List


def find_splits(maps_path: Path) -> List[int]:
    """Find which splits that were trained in the MAPS."""
    splits = [
        int(split.name.split("-")[1])
        for split in list(maps_path.iterdir())
        if split.name.startswith("split-")
    ]
    return splits


def find_stopped_splits(maps_path: Path) -> List[int]:
    """Find which splits for which training was not completed."""
    existing_split_list = find_splits(maps_path)
    stopped_splits = []
    for split in existing_split_list:
        for dir in list((maps_path / f"split-{split}").iterdir()):
            if (
                maps_path / f"split-{split}" / "tmp"
            ).is_dir() or "best-" not in dir.name:
                stopped_splits.append(split)

    return stopped_splits


def find_finished_splits(maps_path: Path) -> List[int]:
    """Find which splits for which training was completed."""
    finished_splits = list()
    existing_split_list = find_splits(maps_path)
    stopped_splits = find_stopped_splits(maps_path)
    for split in existing_split_list:
        if split not in stopped_splits:
            performance_dir_list = [
                performance_dir
                for performance_dir in list((maps_path / f"split-{split}").iterdir())
                if "best-" in performance_dir.name
            ]
            if len(performance_dir_list) > 0:
                finished_splits.append(split)
    return finished_splits


def print_description_log(
    maps_path: Path,
    data_group: str,
    split: int,
    selection_metric: str,
):
    """
    Print the description log associated to a prediction or interpretation.

    Args:
        data_group (str): name of the data group used for the task.
        split (int): Index of the split used for training.
        selection_metric (str): Metric used for best weights selection.
    """
    log_dir = maps_path / f"split-{split}" / f"best-{selection_metric}" / data_group
    log_path = log_dir / "description.log"
    with log_path.open(mode="r") as f:
        content = f.read()
