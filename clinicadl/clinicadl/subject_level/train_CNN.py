import argparse
import torch
import sys
from time import time
from os import path
from torch.utils.data import DataLoader

from .utils import train
from tools.deep_learning.data import MinMaxNormalization, MRIDataset, load_data
from tools.deep_learning import create_model, commandline_to_json
from tools.deep_learning.models import transfer_learning

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to log dir for tensorboard usage.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("model", type=str,
                    help="model selected")

# Data Management
parser.add_argument("--preprocessing", default="linear", choices=["linear", "mni"], type=str,
                    help="Defines the path to data in CAPS.")
parser.add_argument("--diagnoses", "-d", default=['AD', 'CN'], nargs='+', type=str,
                    help="The diagnoses used for the classification")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization.")

# Cross-validation
parser.add_argument("--n_splits", type=int, default=None,
                    help="If a value is given will load data of a k-fold CV")
parser.add_argument("--split", type=int, default=0,
                    help="Will load the specific split wanted.")

# Training arguments
parser.add_argument('--accumulation_steps', '-asteps', default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--patience", type=int, default=10,
                    help="Waiting time for early stopping.")
parser.add_argument("--tolerance", type=float, default=0.05,
                    help="Tolerance value for the early stopping.")

# Transfer learning arguments
parser.add_argument("--transfer_learning_path", default=None, type=str,
                    help="If an existing path is given, a pretrained autoencoder is used.")
parser.add_argument("--selection", default="best_acc", choices=["best_loss", "best_acc"], type=str,
                    help="Allow to choose which model of the experiment is loaded.")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay of the optimizer')


# Computational issues
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()

    text_file = open(path.join(options.output_dir, 'python_version.txt'), 'w')
    text_file.write('Version of python: %s \n' % sys.version)
    text_file.write('Version of pytorch: %s \n' % torch.__version__)
    text_file.close()

    # Get the data.
    training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                        options.split, options.n_splits, options.baseline)

    data_train = MRIDataset(options.input_dir, training_tsv, options.preprocessing, transform=transformations)
    data_valid = MRIDataset(options.input_dir, valid_tsv, options.preprocessing, transform=transformations)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=options.batch_size,
                              shuffle=True,
                              num_workers=options.num_workers,
                              pin_memory=True
                              )

    valid_loader = DataLoader(data_valid,
                              batch_size=options.batch_size,
                              shuffle=False,
                              num_workers=options.num_workers,
                              pin_memory=True
                              )

    # Initialize the model
    print('Initialization of the model')
    model = create_model(options.model, options.gpu)
    model = transfer_learning(model, options.split, options.output_dir, options.transfer_learning_path, options.gpu,
                              selection=options.selection)

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                         options.learning_rate, weight_decay=options.weight_decay)

    print('Beginning the training task')
    train(model, train_loader, valid_loader, criterion, optimizer, False, options)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, 'CNN')
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
