# coding: utf8
from clinicadl import MapsManager


def train(maps_dir, train_dict, split, erase_existing=True):

    maps_manager = MapsManager(maps_dir, train_dict, verbose="info")
    maps_manager.train(split=split, overwrite=erase_existing)
