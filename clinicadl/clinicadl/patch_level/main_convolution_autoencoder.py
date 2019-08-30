import argparse
import copy
import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from .utils import MRIDataset_patch_hippocampus, MRIDataset_patch
from .utils import stacked_ae_learning, visualize_ae

from tools.deep_learning import commandline_to_json, create_model
from tools.deep_learning.data import load_data, MinMaxNormalization

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for 3D convolutional autoencoder, the AE will be reconstructed based on the CNN that you choose")

# Mandatory arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to tsv file of the population based on the diagnosis tsv files."
                         "To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs and the tsv files containing the performances.")

# Data management
parser.add_argument("--diagnoses", default=["AD", "CN", "MCI"], type=str, nargs="+",
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument('--hippocampus_roi', default=False, action='store_true',
                    help="If train the model using only hippocampus ROI")

# Cross-validation
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Default behaviour will run all splits, else only the splits specified will be run.")

# Training arguments
parser.add_argument("--network", default="Conv4_FC3",
                    help="Architecture of the network on which the autoencoder is based.")
parser.add_argument("--epochs", default=1, type=int,
                    help="Epochs through the data.")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")

# visualization
parser.add_argument("--visualization", default=True, type=bool,
                    help='Chooses if visualization is done on AE pretraining')

# Computational issues
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--gpu", default=False, action='store_true',
                    help='Uses gpu instead of cpu if cuda is available')


def main(options):

    model = create_model(options.network, options.gpu)
    init_state = copy.deepcopy(model.state_dict())
    transformations = transforms.Compose([MinMaxNormalization()])

    if options.split is None:
        fold_iterator = range(options.n_splits)
    else:
        fold_iterator = [options.split]

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(options.diagnosis_tsv_path, options.diagnoses, fi,
                                            n_splits=options.n_splits, baseline=options.baseline)

        print("Running for the %d -th fold" % fi)

        if options.hippocampus_roi:
            print("Only using hippocampus ROI")

            data_train = MRIDataset_patch_hippocampus(options.caps_directory, training_tsv, transformations=transformations)
            data_valid = MRIDataset_patch_hippocampus(options.caps_directory, valid_tsv, transformations=transformations)

        else:
            data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size,
                                          options.patch_stride, transformations=transformations)
            data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size,
                                          options.patch_stride, transformations=transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=True,
                                  num_workers=options.num_workers,
                                  drop_last=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        model.load_state_dict(init_state)

        # Decide to use gpu or cpu to train the autoencoder
        if options.gpu == False:
            # example image for tensorbordX usage:$
            example_batch = (next(iter(train_loader))['image'])[0, ...].unsqueeze(0)
        else:
            print("Using GPU")
            # example image for tensorbordX usage:$
            example_batch = (next(iter(train_loader))['image'].cuda())[0, ...].unsqueeze(0)

        criterion = torch.nn.MSELoss()
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "valid")))

        model, best_autodecoder = stacked_ae_learning(model, train_loader, valid_loader, criterion, writer_train,
                                                      writer_valid, options, fi)

        if options.visualization:
            visualize_ae(best_autodecoder, example_batch, os.path.join(options.output_dir, "visualize", "fold_" + str(fi)))

        del best_autodecoder, train_loader, valid_loader, example_batch, criterion
        torch.cuda.empty_cache()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, "ConvAutoencoder")
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
