# coding: utf8
from pathlib import Path
from typing import Any, Dict, List

from clinicadl import MapsManager


def train(
    maps_dir: Path,
    train_dict: Dict[str, Any],
    split_list: List[int],
    erase_existing: bool = True,
):

    from codecarbon import EmissionsTracker

    tracker = EmissionsTracker()
    tracker.start()
    maps_manager = MapsManager(maps_dir, train_dict, verbose=None)
    maps_manager.train(split_list=split_list, overwrite=erase_existing)
    tracker.stop()
