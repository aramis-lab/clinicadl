# coding: utf8

import torch
import os
from torch.utils.data import DataLoader

from ..tools.deep_learning.autoencoder_utils import train, visualize_image, get_criterion
from ..tools.deep_learning.models import init_model, load_model, transfer_learning
from ..tools.deep_learning.data import (load_data,
                                        get_transforms,
                                        return_dataset,
                                        generate_sampler)
from ..tools.deep_learning.iotools import return_logger, check_and_clean
from ..tools.deep_learning.iotools import commandline_to_json, write_requirements_version, translate_parameters


def train_autoencoder(params):
    """
    Trains an autoencoder and writes:
        - logs obtained with Tensorboard during training,
        - best models obtained according to the validation loss,
        - for patch and roi modes, the initialization state is saved as it is identical across all folds,
        - autoencoder reconstructions in nifti files at the end of the training.

    If the training crashes it is possible to relaunch the training process from the checkpoint.pth.tar and
    optimizer.pth.tar files which respectively contains the state of the model and the optimizer at the end
    of the last epoch that was completed before the crash.
    """
    main_logger = return_logger(params.verbose, "main process")
    train_logger = return_logger(params.verbose, "train")
    check_and_clean(params.output_dir)

    commandline_to_json(params, logger=main_logger)
    write_requirements_version(params.output_dir)
    params = translate_parameters(params)

    train_transforms, all_transforms = get_transforms(params.mode,
                                                      minmaxnormalization=params.minmaxnormalization,
                                                      data_augmentation=params.data_augmentation)
    criterion = get_criterion(params.loss)

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
            multi_cohort=params.multi_cohort
        )

        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    train_transformations=train_transforms, all_transformations=all_transforms,
                                    prepare_dl=params.prepare_dl, multi_cohort=params.multi_cohort,
                                    params=params)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    train_transformations=train_transforms, all_transformations=all_transforms,
                                    prepare_dl=params.prepare_dl, multi_cohort=params.multi_cohort,
                                    params=params)

        train_sampler = generate_sampler(data_train, params.sampler)

        train_loader = DataLoader(
                data_train,
                batch_size=params.batch_size,
                sampler=train_sampler,
                num_workers=params.num_workers,
                pin_memory=True)

        valid_loader = DataLoader(
                data_valid,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True)

        # Define output directories
        log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models')
        visualization_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'autoencoder_reconstruction')

        decoder = init_model(params, initial_shape=data_train.size, autoencoder=True)
        decoder = transfer_learning(decoder, fi, source_path=params.transfer_learning_path,
                                    gpu=params.gpu, selection=params.transfer_learning_selection)
        optimizer = getattr(torch.optim, params.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                           lr=params.learning_rate,
                                                           weight_decay=params.weight_decay)

        train(decoder, train_loader, valid_loader, criterion, optimizer, False,
              log_dir, model_dir, params, train_logger)

        if params.visualization:
            best_decoder, _ = load_model(decoder, os.path.join(model_dir, "best_loss"),
                                         params.gpu, filename='model_best.pth.tar')
            nb_images = data_train.size.elem_per_image
            if nb_images <= 2:
                nb_images *= 3
            visualize_image(best_decoder, valid_loader, os.path.join(visualization_dir, "validation"),
                            nb_images=nb_images)
            visualize_image(best_decoder, train_loader, os.path.join(visualization_dir, "train"),
                            nb_images=nb_images)
        del decoder
        torch.cuda.empty_cache()
