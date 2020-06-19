# coding: utf8

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
import torch
import os
from time import time

from ..tools.deep_learning import create_model
from ..tools.deep_learning.data import (load_data,
                                        MinMaxNormalization,
                                        MRIDataset_slice)
from ..tools.deep_learning.cnn_utils import train
from ..patch_level.evaluation_singleCNN import test_cnn

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018-2020 The Aramis Lab Team"
__credits__ = ["Junhao Wen" "Elina Thibeau-Sutre" "Mauricio Diaz"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen, Elina Thibeau-Sutre, Mauricio Diaz"
__email__ = "junhao.wen89@gmail.com, mauricio.diaz@inria.fr"
__status__ = "Development"

# Train 2D CNN - Slice level network
# The input MRI's dimension is 169*208*179 after cropping"


def train_slice(params):

    # Initialize the model
    print('Do transfer learning with existed model trained on ImageNet!\n')
    print('The chosen network is %s !' % params.model)

    model = create_model(params.model, params.gpu, dropout=params.dropout)
    trg_size = (224, 224)  # most of the imagenet pretrained model has this input size

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
    # images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in
    # to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])

    # calculate the time consummation
    total_time = time()
    init_state = copy.deepcopy(model.state_dict())

    if params.split is None:
        fold_iterator = range(params.n_splits)
    else:
        fold_iterator = params.split

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(params.tsv_path, params.diagnoses, fi,
                                            n_splits=params.n_splits, baseline=params.baseline)

        print("Running for the %d-th fold" % fi)

        data_train = MRIDataset_slice(params.input_dir, training_tsv, transformations=transformations,
                                      preprocessing=params.preprocessing, mri_plane=params.mri_plane,
                                      prepare_dl=params.prepare_dl,
                                      discarded_slices=params.discarded_slices)
        data_valid = MRIDataset_slice(params.input_dir, valid_tsv, transformations=transformations,
                                      preprocessing=params.preprocessing, mri_plane=params.mri_plane,
                                      prepare_dl=params.prepare_dl,
                                      discarded_slices=params.discarded_slices)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=params.batch_size,
                                  shuffle=True,
                                  num_workers=params.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(data_valid,
                                  batch_size=params.batch_size,
                                  shuffle=False,
                                  num_workers=params.num_workers,
                                  pin_memory=True)

        # Initialize the model
        print('Initialization of the model')
        model.load_state_dict(init_state)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + params.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                            lr=params.learning_rate,
                                                            weight_decay=params.weight_decay)
        setattr(params, 'beginning_epoch', 0)

        # Define output directories
        log_dir = os.path.join(params.output_dir, "log_dir", "fold_%i" % fi, "CNN")
        model_dir = os.path.join(params.output_dir, "best_model_dir", "fold_%i" % fi, "CNN")

        print('Beginning the training task')
        train(model, train_loader, valid_loader, criterion, optimizer, False, log_dir, model_dir, params)

        test_cnn(train_loader, "train", fi, criterion, params)
        test_cnn(valid_loader, "validation", fi, criterion, params)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
