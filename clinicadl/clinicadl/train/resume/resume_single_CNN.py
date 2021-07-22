# coding: utf8

from os import path

from torch.utils.data import DataLoader

from clinicadl.train.train_singleCNN import test_single_cnn
from clinicadl.utils.caps_dataset.data import (
    generate_sampler,
    get_transforms,
    load_data,
    return_dataset,
)
from clinicadl.utils.maps_manager.iotools import (
    commandline_to_json,
    return_logger,
    translate_parameters,
    write_requirements_version,
)
from clinicadl.utils.network.cnn_utils import get_criterion, train
from clinicadl.utils.network.models import init_model, load_model, load_optimizer


def resume_single_cnn(params, resumed_split):
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")
    eval_logger = return_logger(params.verbose, "final evaluation")

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
    model = init_model(
        params, initial_shape=data_train.size, len_atlas=data_train.len_atlas()
    )
    model_dir = path.join(params.output_dir, f"fold-{resumed_split}", "models")
    model, current_epoch = load_model(
        model, model_dir, params.gpu, "checkpoint.pth.tar"
    )

    params.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = get_criterion(params.loss)
    optimizer_path = path.join(
        params.output_dir, f"fold-{resumed_split}", "models", "optimizer.pth.tar"
    )
    optimizer = load_optimizer(optimizer_path, model)

    # Define output directories
    log_dir = path.join(params.output_dir, f"fold-{resumed_split}", "tensorboard_logs")
    model_dir = path.join(params.output_dir, f"fold-{resumed_split}", "models")

    main_logger.debug("Beginning the training task")
    train(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        True,
        log_dir,
        model_dir,
        params,
        train_logger,
    )

    test_single_cnn(
        model,
        params.output_dir,
        train_loader,
        "train",
        resumed_split,
        criterion,
        params.mode,
        eval_logger,
        params.selection_threshold,
        gpu=params.gpu,
    )
    test_single_cnn(
        model,
        params.output_dir,
        valid_loader,
        "validation",
        resumed_split,
        criterion,
        params.mode,
        eval_logger,
        params.selection_threshold,
        gpu=params.gpu,
    )
