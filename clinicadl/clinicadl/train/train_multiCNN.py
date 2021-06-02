# coding: utf8

import os

import torch
from torch.utils.data import DataLoader

from ..tools.deep_learning.cnn_utils import (
    get_criterion,
    mode_level_to_tsvs,
    soft_voting_to_tsvs,
    test,
    train,
)
from ..tools.deep_learning.data import (
    compute_num_cnn,
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
from ..tools.deep_learning.models import create_model, load_model, transfer_learning


def train_multi_cnn(params, erase_existing=True):
    """
    Trains one CNN per patch and writes for each CNN:
        - logs obtained with Tensorboard during training,
        - best models obtained according to two metrics on the validation set (loss and balanced accuracy),
        - final performances at the end of the training.
    Performances are also aggregated at the image level and combines the output of all networks.
    The initialization state is shared across all networks.

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

    num_cnn = compute_num_cnn(params.input_dir, params.tsv_path, params, data="train")

    if num_cnn == 1:
        raise ValueError(
            "Multi-CNN framework cannot be performed on a dataset computing one element per image."
        )

    if params.split is None:
        if params.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    # Find main metric
    if params.task == "classification":
        main_metric = "balanced_accuracy"
    else:
        main_metric = "mae"

    # Loop on folds
    for fi in fold_iterator:
        main_logger.info(f"Fold {fi}")

        for cnn_index in range(num_cnn):

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
                cnn_index=cnn_index,
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
                cnn_index=cnn_index,
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
            main_logger.info(f"Initialization of the model {cnn_index}")
            model = create_model(
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
                {
                    "n_classes": data_train.n_classes(),
                    "label_code": data_train.label_code,
                },
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
            log_dir = os.path.join(
                params.output_dir,
                f"fold-{fi}",
                "tensorboard_logs",
                f"cnn-{cnn_index}",
            )
            model_dir = os.path.join(
                params.output_dir, f"fold-{fi}", "models", f"cnn-{cnn_index}"
            )

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

            test_cnn(
                model,
                params.output_dir,
                train_loader,
                "train",
                fi,
                criterion,
                cnn_index,
                mode=params.mode,
                gpu=params.gpu,
                task=params.network_task,
                logger=eval_logger,
            )
            test_cnn(
                model,
                params.output_dir,
                valid_loader,
                "validation",
                fi,
                criterion,
                cnn_index,
                mode=params.mode,
                gpu=params.gpu,
                task=params.network_task,
                logger=eval_logger,
            )

        for selection in [f"best_{main_metric}", "best_loss"]:
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                logger=eval_logger,
                mode=params.mode,
                dataset="train",
                num_cnn=num_cnn,
                selection_threshold=params.selection_threshold,
                task=params.network_task,
            )
            soft_voting_to_tsvs(
                params.output_dir,
                fi,
                selection,
                logger=eval_logger,
                mode=params.mode,
                dataset="validation",
                num_cnn=num_cnn,
                selection_threshold=params.selection_threshold,
                task=params.network_task,
            )


def test_cnn(
    model,
    output_dir,
    data_loader,
    subset_name,
    split,
    criterion,
    cnn_index,
    mode,
    logger,
    task,
    gpu=False,
):
    if task == "classification":
        main_metric = "balanced_accuracy"
    else:
        main_metric = "mae"

    for selection in [f"best_{main_metric}", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(
            model,
            os.path.join(
                output_dir, f"fold-{split}", "models", f"cnn-{cnn_index}", selection
            ),
            gpu=gpu,
            filename="model_best.pth.tar",
        )

        results_df, metrics = test(model, data_loader, gpu, criterion, mode, task=task)

        # logger.info(
        #     f"{subset_name} balanced accuracy is {metrics['balanced_accuracy']} "
        #     f"for {mode} {cnn_index} and model selected on {selection}"
        # )

        mode_level_to_tsvs(
            output_dir,
            results_df,
            metrics,
            split,
            selection,
            mode,
            dataset=subset_name,
            cnn_index=cnn_index,
        )
