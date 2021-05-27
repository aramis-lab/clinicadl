# coding: utf8

from os import path

import torch
from torch.utils.data import DataLoader

from ..tools.deep_learning.autoencoder_utils import (
    get_criterion,
    train,
    visualize_image,
)
from ..tools.deep_learning.data import (
    generate_sampler,
    get_transforms,
    load_data,
    return_dataset,
)
from ..tools.deep_learning.iotools import (
    commandline_to_json,
    return_logger,
    translate_parameters,
    write_requirements_version,
)
from ..tools.deep_learning.models import init_model, load_model, load_optimizer


def resume_autoencoder(params, resumed_split):
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")

    commandline_to_json(params, logger=main_logger)
    write_requirements_version(params.output_dir)
    params = translate_parameters(params)
    train_transforms, all_transforms = get_transforms(
        params.mode,
        minmaxnormalization=params.minmaxnormalization,
        data_augmentation=params.data_augmentation,
    )

    training_df, valid_df = load_data(
        params.tsv_path,
        params.diagnoses,
        resumed_split,
        n_splits=params.n_splits,
        baseline=params.baseline,
        logger=main_logger,
        multi_cohort=params.multi_cohort,
    )

    data_train = return_dataset(
        params.mode,
        params.input_dir,
        training_df,
        params.preprocessing,
        train_transformations=train_transforms,
        all_transformations=all_transforms,
        params=params,
    )
    data_valid = return_dataset(
        params.mode,
        params.input_dir,
        valid_df,
        params.preprocessing,
        train_transformations=train_transforms,
        all_transformations=all_transforms,
        params=params,
    )

    train_sampler = generate_sampler(data_train, params.sampler)

    train_loader = DataLoader(
        data_train,
        batch_size=params.batch_size,
        sampler=train_sampler,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        data_valid,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
    )

    # Initialize the model
    main_logger.info("Initialization of the model")
    decoder = init_model(params, initial_shape=data_train.size, autoencoder=True)
    model_dir = path.join(params.output_dir, f"fold-{resumed_split}", "models")
    decoder, current_epoch = load_model(
        decoder, model_dir, params.gpu, "checkpoint.pth.tar"
    )

    params.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = get_criterion(params.loss)
    optimizer_path = path.join(
        params.output_dir, f"fold-{resumed_split}", "models", "optimizer.pth.tar"
    )
    optimizer = load_optimizer(optimizer_path, decoder)

    # Define output directories
    log_dir = path.join(params.output_dir, f"fold-{resumed_split}", "tensorboard_logs")
    model_dir = path.join(params.output_dir, f"fold-{resumed_split}", "models")
    visualization_dir = path.join(
        params.output_dir, f"fold-{resumed_split}", "autoencoder_reconstruction"
    )

    main_logger.debug("Beginning the training task")
    train(
        decoder,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        False,
        log_dir,
        model_dir,
        params,
        train_logger,
    )

    if params.visualization:
        best_decoder, _ = load_model(
            decoder,
            path.join(model_dir, "best_loss"),
            params.gpu,
            filename="model_best.pth.tar",
        )
        nb_images = data_train.size.elem_per_image
        if nb_images <= 2:
            nb_images *= 3
        visualize_image(
            best_decoder,
            valid_loader,
            path.join(visualization_dir, "validation"),
            nb_images=nb_images,
        )
        visualize_image(
            best_decoder,
            train_loader,
            path.join(visualization_dir, "train"),
            nb_images=nb_images,
        )
    del decoder
    torch.cuda.empty_cache()
