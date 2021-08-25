# coding: utf8
from clinicadl import MapsManager


def train(maps_dir, train_dict, folds, erase_existing=True):

    maps_manager = MapsManager(maps_dir, train_dict, verbose="info")
    maps_manager.train(folds=folds, overwrite=erase_existing)
