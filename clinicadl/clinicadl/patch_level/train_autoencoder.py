# coding: utf8

import torch
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ..tools.deep_learning.autoencoder_utils import train
from ..tools.deep_learning import create_autoencoder
from ..tools.deep_learning.data import (load_data,
                                        MinMaxNormalization,
                                        MRIDataset_patch,
                                        MRIDataset_patch_hippocampus)


def train_autoencoder_patch(params):

    transformations = transforms.Compose([MinMaxNormalization()])
    criterion = torch.nn.MSELoss()

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(
                params.tsv_path,
                params.diagnoses,
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline
                )

        print("Running for the %d-th fold" % fi)

        if params.hippocampus_roi:
            print("Only using hippocampus ROI")

            data_train = MRIDataset_patch_hippocampus(
                params.input_dir,
                training_tsv,
                preprocessing=params.preprocessing,
                transformations=transformations
            )
            data_valid = MRIDataset_patch_hippocampus(
                params.input_dir,
                valid_tsv,
                preprocessing=params.preprocessing,
                transformations=transformations
            )

        else:
            data_train = MRIDataset_patch(
                    params.input_dir,
                    training_tsv,
                    params.patch_size,
                    params.stride_size,
                    preprocessing=params.preprocessing,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl)
            data_valid = MRIDataset_patch(
                    params.input_dir,
                    valid_tsv,
                    params.patch_size,
                    params.stride_size,
                    preprocessing=params.preprocessing,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl)

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
        log_dir = os.path.join(params.output_dir, "log_dir", "fold_%i" % fi, "ConvAutoencoder")
        model_dir = os.path.join(params.output_dir, "best_model_dir", "fold_%i" % fi, "ConvAutoencoder")
        visualization_dir = os.path.join(params.output_dir, 'visualize', 'fold_%i' % fi)

        # Hard-coded arguments for patch
        setattr(params, "accumulation_steps", 1)
        setattr(params, "evaluation_steps", 0)

        decoder = create_autoencoder(params.model, gpu=params.gpu)
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, decoder.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)

        train(decoder, train_loader, valid_loader, criterion, optimizer, False,
              log_dir, model_dir, visualization_dir, params)
        del decoder
        torch.cuda.empty_cache()

