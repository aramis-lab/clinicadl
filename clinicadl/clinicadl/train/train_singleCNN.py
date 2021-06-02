# coding: utf8

import os

import torch
from torch.utils.data import DataLoader

from ..tools.deep_learning.cnn_utils import (
    get_criterion,
    mode_level_to_tsvs,
    mode_to_image_tsvs,
    soft_voting_to_tsvs,
    test,
    train,
)
from ..tools.deep_learning.data import (
    generate_sampler,
    get_transforms,
    load_data,
    return_dataset,
)
from ..tools.deep_learning.iotools import (
    append_to_json,
    check_and_clean,
    commandline_to_json,
    return_logger,
    translate_parameters,
    write_requirements_version,
)
from ..tools.deep_learning.models import init_model, load_model, transfer_learning


def train_single_cnn(params, erase_existing=True):
    """
    Trains a single CNN and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - final performances at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")
    eval_logger = return_logger(params.verbose, "final evaluation")
    if erase_existing:
        check_and_clean(params.output_dir)

    commandline_to_json(params, logger=main_logger)
    write_requirements_version(params.output_dir)
    params = translate_parameters(params)
    train_transforms, all_transforms = get_transforms(
        params.mode,
        minmaxnormalization=params.minmaxnormalization,
        data_augmentation=params.data_augmentation,
    )

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:
        main_logger.info("Fold %i" % fi)

        training_df, valid_df = load_data(
            params.tsv_path,
            params.diagnoses,
            fi,
            n_splits=params.n_splits,
            baseline=params.baseline,
            logger=main_logger,
            multi_cohort=params.multi_cohort,
        )

        data_train = return_dataset(
            params.mode,
            params.input_dir,
            training_df,
            preprocessing=params.preprocessing,
            label=params.label,
            task=params.network_task,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
            prepare_dl=params.prepare_dl,
            multi_cohort=params.multi_cohort,
            params=params,
        )
        data_valid = return_dataset(
            params.mode,
            params.input_dir,
            valid_df,
            preprocessing=params.preprocessing,
            label=params.label,
            task=params.network_task,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
            prepare_dl=params.prepare_dl,
            multi_cohort=params.multi_cohort,
            params=params,
        )
        if (
            data_train.label_code is not None
            and data_train.label_code != data_valid.label_code
        ):
            raise ValueError(
                "The label codes computed from training and validation data are different. "
                "Please ensure that your classes are all represented in training and validation data. "
                f"{data_train.label_code} != {data_valid.label_code}"
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
            params, initial_shape=data_train.size, n_classes=data_train.n_classes()
        )
        model = transfer_learning(
            model,
            fi,
            source_path=params.transfer_learning_path,
            gpu=params.gpu,
            selection=params.transfer_learning_selection,
            logger=main_logger,
        )
        # Save number of classes for other functionalities
        append_to_json(
            {"n_classes": data_train.n_classes(), "label_code": data_train.label_code},
            params,
        )

        # Define criterion and optimizer
        criterion = get_criterion(params.network_task)
        optimizer = getattr(torch.optim, params.optimizer)(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=params.learning_rate,
            weight_decay=params.weight_decay,
        )

        # Define output directories
        log_dir = os.path.join(params.output_dir, f"fold-{fi}", "tensorboard_logs")
        model_dir = os.path.join(params.output_dir, f"fold-{fi}", "models")

        main_logger.debug("Beginning the training task")
        train(
            model,
            train_loader,
            valid_loader,
            criterion,
            optimizer,
            False,
            log_dir,
            model_dir,
            params,
            params.network_task,
            logger=train_logger,
        )

        test_single_cnn(
            model,
            params.output_dir,
            train_loader,
            "train",
            fi,
            criterion,
            params.mode,
            eval_logger,
            params.selection_threshold,
            task=params.network_task,
            gpu=params.gpu,
        )
        test_single_cnn(
            model,
            params.output_dir,
            valid_loader,
            "validation",
            fi,
            criterion,
            params.mode,
            eval_logger,
            params.selection_threshold,
            task=params.network_task,
            gpu=params.gpu,
        )


def test_single_cnn(
    model,
    output_dir,
    data_loader,
    subset_name,
    split,
    criterion,
    mode,
    logger,
    selection_threshold,
    task,
    gpu=False,
):

    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(
            model,
            os.path.join(output_dir, f"fold-{split}", "models", selection),
            gpu=gpu,
            filename="model_best.pth.tar",
        )

        results_df, metrics = test(model, data_loader, gpu, criterion, mode, task=task)
        # logger.info(
        #     f"{mode} level {subset_name} balanced accuracy is {metrics['balanced_accuracy']} "
        #     f"for model selected on {selection}"
        # )

        mode_level_to_tsvs(
            output_dir, results_df, metrics, split, selection, mode, dataset=subset_name
        )

        # Soft voting
        if data_loader.dataset.elem_per_image > 1:
            soft_voting_to_tsvs(
                output_dir,
                split,
                logger=logger,
                selection=selection,
                mode=mode,
                dataset=subset_name,
                selection_threshold=selection_threshold,
                task=task,
            )
        elif mode != "image":
            mode_to_image_tsvs(
                output_dir, split, selection=selection, mode=mode, dataset=subset_name
            )
