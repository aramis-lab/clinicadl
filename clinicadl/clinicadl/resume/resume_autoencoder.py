# coding: utf8
import torch
import os
from os import path
from torch.utils.data import DataLoader
from warnings import warn

from clinicadl.tools.deep_learning.data import return_dataset, get_transforms, load_data
from clinicadl.tools.deep_learning import load_model, create_autoencoder, load_optimizer, read_json
from clinicadl.tools.deep_learning.autoencoder_utils import train, visualize_image


def resume_autoencoder(params):

    params = read_json(params)

    if params.evaluation_steps % params.accumulation_steps != 0 and params.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (params.evaluation_steps, params.accumulation_steps))

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

        log_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'tensorboard_logs')
        model_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'models')
        visualization_dir = os.path.join(params.output_dir, 'fold-%i' % fi, 'autoencoder_reconstruction')

        decoder = create_autoencoder(params.model, gpu=params.gpu)  # TODO Change here after merging #71
        decoder, current_epoch, num_bad_epochs = load_model(
            decoder,
            model_dir,
            gpu=params.gpu,
            filename='checkpoint.pth.tar'
        )
        optimizer_path = path.join(model_dir, 'optimizer.pth.tar')
        optimizer = load_optimizer(optimizer_path, decoder)

        # Prepare args for resuming
        params.beginning_epoch = current_epoch + 1
        if num_bad_epochs is None:
            warn("You are resuming a job conducted with an older version of clinicadl."
                 "Information on patience for early stopping was lost.")
            params.num_bad_epochs = 0
        else:
            params.num_bad_epochs = num_bad_epochs

        print('Resuming the training task')
        train(decoder, train_loader, valid_loader, criterion, optimizer, True,
              log_dir, model_dir, params)

        if params.visualization:
            print("Visualization of autoencoder reconstruction")
            best_decoder, _, _ = load_model(decoder, os.path.join(model_dir, "best_loss"),
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
