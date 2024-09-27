from pathlib import Path
from typing import List, Optional

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


def print_description_log(
    maps_path: Path,
    split_name: str,
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
    log_dir = (
        maps_path / f"{split_name}-{split}" / f"best-{selection_metric}" / data_group
    )
    log_path = log_dir / "description.log"
    with log_path.open(mode="r") as f:
        content = f.read()


def init_split_manager(
    validation,
    parameters,
    split_list=None,
    ssda_bool: bool = False,
    caps_target: Optional[Path] = None,
    tsv_target_lab: Optional[Path] = None,
):
    from clinicadl.validation import split_manager

    split_class = getattr(split_manager, validation)
    args = list(
        split_class.__init__.__code__.co_varnames[
            : split_class.__init__.__code__.co_argcount
        ]
    )
    args.remove("self")
    args.remove("split_list")
    kwargs = {"split_list": split_list}
    for arg in args:
        kwargs[arg] = parameters[arg]

    if ssda_bool:
        kwargs["caps_directory"] = caps_target
        kwargs["tsv_path"] = tsv_target_lab

    return split_class(**kwargs)
