"""
Automatic relaunch of jobs that were stopped before the end of training.
Unfinished folds are detected as they do not contain a "performances" sub-folder
"""

import argparse
import os
from os import path


def replace_arg(options, key_name, value):
    if value is not None:
        setattr(options, key_name, value)


def automatic_resume(
    model_path, gpu, batch_size, num_workers, evaluation_steps, verbose=0
):
    from clinicadl.utils.maps_manager.iotools import read_json, return_logger

    from ..train_autoencoder import train_autoencoder
    from ..train_multiCNN import train_multi_cnn
    from ..train_singleCNN import train_single_cnn
    from .resume_autoencoder import resume_autoencoder
    from .resume_single_CNN import resume_single_cnn

    logger = return_logger(verbose=verbose, name_fn="automatic resume")

    options = argparse.Namespace()
    options.model_path = model_path
    logger.info(f"Job being resumed: {model_path}")

    options = read_json(options, read_computational=True)

    # Set computational parameters
    replace_arg(options, "gpu", gpu)
    replace_arg(options, "batch_size", batch_size)
    replace_arg(options, "num_workers", num_workers)
    replace_arg(options, "evaluation_steps", evaluation_steps)

    # Set verbose
    options.verbose = verbose

    fold_list = sorted(
        [
            int(fold.split("-")[1])
            for fold in os.listdir(options.model_path)
            if fold[:4:] == "fold"
        ]
    )
    finished_folds = [
        fold
        for fold in fold_list
        if "cnn_classification"
        in os.listdir(path.join(options.model_path, f"fold-{fold}"))
    ]
    stopped_folds = [
        fold
        for fold in fold_list
        if fold not in finished_folds
        and "checkpoint.pth.tar"
        in os.listdir(path.join(options.model_path, f"fold-{fold}", "models"))
    ]

    if options.split is None:
        if options.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(options.n_splits)
    else:
        fold_iterator = options.split

    absent_folds = [
        fold
        for fold in fold_iterator
        if fold not in finished_folds and fold not in stopped_folds
    ]
    logger.info(f"Finished folds {finished_folds}")
    logger.info(f"Stopped folds {stopped_folds}")
    logger.info(f"Missing folds {absent_folds}")

    # To ensure retro-compatibility with random search
    options.output_dir = options.model_path

    for fold in stopped_folds:
        if options.network_type == "cnn":
            resume_single_cnn(options, fold)
        elif options.network_type == "autoencoder":
            resume_autoencoder(options, fold)
        else:
            raise NotImplementedError(
                f"Resume function is not implemented for network type {options.network_type}"
            )

    if len(absent_folds) != 0:
        options.split = absent_folds
        if options.network_type == "cnn":
            train_single_cnn(options, erase_existing=False)
        elif options.network_type == "multicnn":
            train_multi_cnn(options, erase_existing=False)
        elif options.network_type == "autoencoder":
            train_autoencoder(options, erase_existing=False)
        else:
            raise NotImplementedError(
                f"Resume function is not implemented for network type {options.network_type}"
            )
