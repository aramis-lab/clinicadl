import argparse
import copy
import torch
import os
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

import sys
from os import path

package_path = path.abspath(path.join(path.abspath(path.join(path.abspath(path.join(path.abspath(path.join(sys.argv[0], os.pardir)), os.pardir)), os.pardir)), os.pardir))
sys.path.append(package_path)

from classifiers.three_d_cnn.patch_level.classification_utils import MRIDataset_patch_hippocampus, MRIDataset_patch
from classifiers.three_d_cnn.patch_level.classification_utils import greedy_layer_wise_learning, stacked_ae_learning, visualize_ae

from tools.deep_learning import commandline_to_json
from tools.deep_learning.data import load_data, MinMaxNormalization
from tools.deep_learning.models import create_model

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for 3D convolutional autoencoder, the AE will be reconstructed based on the CNN that you choose")

# Data arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--data_type", default="from_patch", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Default behaviour will run all splits, else only the splits specified will be run.")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument('--hippocampus_roi', default=False, type=bool,
                    help="If train the model using only hippocampus ROI")

# Training arguments
parser.add_argument("--network", default="Conv_4_FC_3", choices=["Conv_4_FC_3", "Conv_7_FC_2", "Conv_3_FC_2"],
                    help="Autoencoder network type. (default=Conv_4_FC_3)")
parser.add_argument("--ae_training_method", default="stacked_ae", choices=["layer_wise_ae", "stacked_ae"],
                    help="How to train the autoencoder, layer wise or train all AEs together")
parser.add_argument("--diagnoses", default=["AD", "CN", "MCI"], type=str, nargs="+",
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs_layer_wise", default=1, type=int,
                    help="Epochs for layer-wise AE training")
parser.add_argument("--epochs_fine_tuning", default=1, type=int,
                    help="Epochs for fine tuning all the stacked AEs after greedy layer-wise training, or directly train the AEs together")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpu", default=False, action='store_true',
                    help='Uses gpu instead of cpu if cuda is available')

# visualization
parser.add_argument("--visualization", default=True, type=bool,
                    help='Chooses if visualization is done on AE pretraining')


def main(options):

    model = create_model(options.network)

    # need to normalized the value to [0, 1]
    transformations = transforms.Compose([MinMaxNormalization()])

    # the initial model weight and bias
    init_state = copy.deepcopy(model.state_dict())

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
            data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size, options.patch_stride, transformations=transformations,
                                          data_type=options.data_type)
            data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size, options.patch_stride, transformations=transformations,
                                          data_type=options.data_type)

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

        if fi != 0:
            model = eval(options.network)()
        model.load_state_dict(init_state)

        # Decide to use gpu or cpu to train the autoencoder
        if options.gpu == False:
            model.cpu()
            # example image for tensorbordX usage:$
            example_batch = (next(iter(train_loader))['image'])[0, ...].unsqueeze(0)
        else:
            print("Using GPU")
            model.cuda()
            # example image for tensorbordX usage:$
            example_batch = (next(iter(train_loader))['image'].cuda())[0, ...].unsqueeze(0)

        criterion = torch.nn.MSELoss()
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "layer_wise", "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "layer_wise", "valid")))
        writer_train_ft = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "fine_tine", "train")))
        writer_valid_ft = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "fine_tine", "valid")))

        if options.ae_training_method == 'layer_wise_ae':
            model, best_autodecoder = greedy_layer_wise_learning(model, train_loader, valid_loader, criterion, writer_train, writer_valid, writer_train_ft, writer_valid_ft, options, fi)
        else:
            model, best_autodecoder = stacked_ae_learning(model, train_loader, valid_loader, criterion,
                                                          writer_train_ft, writer_valid_ft,
                                                                 options, fi)

        if options.visualization:
            visualize_ae(best_autodecoder, example_batch, os.path.join(options.output_dir, "visualize", "fold_" + str(fi)))

        del best_autodecoder, train_loader, valid_loader, example_batch, criterion, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    commandline_to_json(commandline, "ConvAutoencoder")
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
