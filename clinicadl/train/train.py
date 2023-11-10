# coding: utf8
from pathlib import Path
from typing import Any, Dict, List

from clinicadl import MapsManager
from clinicadl.utils.data_handler.data_config import DataConfig


def train(
    data_config: DataConfig,
    erase_existing: bool = True,
):
    print(data_config)
    maps_manager = MapsManager(data_config, verbose=None)

    print("test pass")
    maps_manager.train(overwrite=erase_existing)
