"""
Retrain a model defined by a commandline.json file
"""

import argparse

from clinicadl.utils.maps_manager.iotools import check_and_complete, read_json

from .launch import train


def retrain(json_path, output_dir, verbose=0):
    options = argparse.Namespace()
    options = read_json(options, json_path=json_path, read_computational=True)
    check_and_complete(options)
    options.output_dir = output_dir
    options.verbose = verbose
    train(options)
