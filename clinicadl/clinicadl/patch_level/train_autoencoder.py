import argparse
import copy
import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from .utils import MRIDataset_patch_hippocampus, MRIDataset_patch
from .utils import stacked_ae_learning, visualize_ae

from ..tools.deep_learning.iotools import Parameters
from ..tools.deep_learning import commandline_to_json, create_model
from ..tools.deep_learning.data import load_data, MinMaxNormalization

__author__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen", "Elina Thibeau-Sutre", "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"


def train_autoencoder_patch(params):

    model = create_model(params.network, params.gpu)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = [params.split]

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(params.tsv_path,
                params.diagnoses, 
                fi,
                n_splits=params.n_splits,
                baseline=params.baseline)

        print("Running for the %d-th fold" % fi)

        if params.hippocampus_roi:
            print("Only using hippocampus ROI")

            data_train = MRIDataset_patch_hippocampus(params.input_dir,
                    training_tsv,
                    transformations=transformations)
            data_valid = MRIDataset_patch_hippocampus(params.input_dir,
                    valid_tsv,
                    transformations=transformations)

        else:
            data_train = MRIDataset_patch(params.input_dir, 
                    training_tsv, 
                    params.patch_size,
                    params.patch_stride,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl)
            data_valid = MRIDataset_patch(params.input_dir, 
                    valid_tsv,
                    params.patch_size,
                    params.patch_stride,
                    transformations=transformations,
                    prepare_dl=params.prepare_dl)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                batch_size=params.batch_size,
                shuffle=True,
                num_workers=params.num_workers,
                pin_memory=True
                )

        valid_loader = DataLoader(data_valid,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True
                )

        model.load_state_dict(init_state)

        criterion = torch.nn.MSELoss()
        writer_train = SummaryWriter(log_dir=(os.path.join(params.output_dir, 
            "log_dir", 
            "fold_" + str(fi),
            "ConvAutoencoder", "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(params.output_dir, 
            "log_dir", 
            "fold_" + str(fi),
            "ConvAutoencoder", "valid")))

        model, best_autodecoder = stacked_ae_learning(model,
                train_loader,
                valid_loader,
                criterion,
                writer_train,
                writer_valid, 
                params, 
                fi)

        if params.visualization:
            example_batch = data_train[0]['image'].unsqueeze(0)
            if params.gpu:
                example_batch = example_batch.cuda()
            visualize_ae(best_autodecoder, example_batch, 
                    os.path.join(params.output_dir, "visualize", "fold_%i" % fi))

        del best_autodecoder, train_loader, valid_loader
        torch.cuda.empty_cache()
