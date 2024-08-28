from pathlib import Path
from typing import List


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
