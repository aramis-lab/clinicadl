# coding: utf8
from pathlib import Path
from typing import Any, Dict, List

from clinicadl import MapsManager
from clinicadl.utils.trainer import Trainer


def train(
    maps_dir: Path,
    train_dict: Dict[str, Any],
    split_list: List[int],
    erase_existing: bool = True,
):
    maps_manager = MapsManager(maps_dir, train_dict, verbose=None)
    trainer = Trainer(maps_manager)
    trainer.train(split_list=split_list, overwrite=erase_existing)
