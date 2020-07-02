# coding: utf8

import torch
import os
from torch.utils.data import DataLoader

from ..tools.deep_learning.autoencoder_utils import train, visualize_image
from ..tools.deep_learning.models import init_model, load_model
from ..tools.deep_learning.data import (load_data,
                                        get_transforms,
                                        return_dataset)


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

    transformations = get_transforms(params.mode, params.minmaxnormalization)
    criterion = torch.nn.MSELoss()

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:

        training_df, valid_df = load_data(
                params.tsv_path,
                params.diagnoses,
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline
                )

        print("Running for the %d-th fold" % fi)

        data_train = return_dataset(params.mode, params.input_dir, training_df, params.preprocessing,
                                    transformations, params)
        data_valid = return_dataset(params.mode, params.input_dir, valid_df, params.preprocessing,
                                    transformations, params)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(
                data_train,
                batch_size=params.batch_size,
                shuffle=True,
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

        decoder = init_model(params.model, gpu=params.gpu, autoencoder=True, dropout=params.dropout)
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)

        train(decoder, train_loader, valid_loader, criterion, optimizer, False,
              log_dir, model_dir, params)

        if params.visualization:
            print("Visualization of autoencoder reconstruction")
            best_decoder, _ = load_model(decoder, os.path.join(model_dir, "best_loss"),
                                         params.gpu, filename='model_best.pth.tar')
            nb_images = train_loader.dataset.elem_per_image
            if nb_images <= 2:
                nb_images *= 3
            visualize_image(best_decoder, valid_loader, os.path.join(visualization_dir, "validation"),
                            nb_images=nb_images)
            visualize_image(best_decoder, train_loader, os.path.join(visualization_dir, "train"),
                            nb_images=nb_images)
        del decoder
        torch.cuda.empty_cache()
