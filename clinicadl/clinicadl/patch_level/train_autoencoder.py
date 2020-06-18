# coding: utf8

import copy
import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from .utils import ae_finetuning, visualize_ae

from ..tools.deep_learning import create_model, load_model
from ..tools.deep_learning.models.autoencoder import AutoEncoder
from ..tools.deep_learning.data import (load_data,
                                        MinMaxNormalization,
                                        MRIDataset_patch,
                                        MRIDataset_patch_hippocampus)


def train_autoencoder_patch(params):

    model = create_model(params.model, params.gpu)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

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

        model.load_state_dict(init_state)

        criterion = torch.nn.MSELoss()

        # Define output directories
        log_dir = os.path.join(params.output_dir, "log_dir", "fold_%i" % fi, "ConvAutoencoder")
        model_dir = os.path.join(params.output_dir, "best_model_dir", "fold_%i" % fi, "ConvAutoencoder")

        writer_train = SummaryWriter(os.path.join(log_dir, "train"))
        writer_valid = SummaryWriter(os.path.join(log_dir, "valid"))

        ae = AutoEncoder(model)
        ae_finetuning(ae, train_loader, valid_loader, criterion, writer_train, writer_valid, params, model_dir)
        best_autodecoder, best_epoch = load_model(ae, os.path.join(model_dir, "best_loss"),
                                                  params.gpu, filename='model_best.pth.tar')
        del ae
        torch.cuda.empty_cache()

        if params.visualization:
            example_batch = data_train[0]['image'].unsqueeze(0)
            if params.gpu:
                example_batch = example_batch.cuda()
            visualize_ae(
                    best_autodecoder,
                    example_batch,
                    os.path.join(
                        params.output_dir,
                        "visualize",
                        "fold_%i" % fi
                        )
                    )

        del best_autodecoder
        torch.cuda.empty_cache()
