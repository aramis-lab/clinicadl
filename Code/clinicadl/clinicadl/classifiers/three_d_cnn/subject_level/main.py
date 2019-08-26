import argparse
import torch
from time import time
from os import path
from torch.utils.data import DataLoader

from classifiers.three_d_cnn.subject_level.classification import train
from tools.deep_learning.data import MinMaxNormalization, MRIDataset, load_data, generate_sampler
from tools.deep_learning import create_model, commandline_to_json

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
parser.add_argument("--preprocessing", default="linear", choices=["linear", "mniskullstrip", "mni"], type=str,
                    help="Defines the path to data in CAPS.")
parser.add_argument("--diagnoses", "-d", default=['AD', 'CN'], nargs='+', type=str,
                    help="The diagnoses used for the classification")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument('--accumulation_steps', '-asteps', default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--test_sessions", default=["ses-M00"], nargs='+', type=str,
                    help="Test the accuracy at the end of the model for the sessions selected")
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization for visualization")
parser.add_argument("--n_splits", type=int, default=None,
                    help="If a value is given will load data of a k-fold CV")
parser.add_argument("--split", type=int, default=0,
                    help="Will load the specific split wanted.")
parser.add_argument("--training_evaluation", default='whole_set', type=str, choices=['whole_set', 'n_batches'],
                    help="Choose the way training evaluation is performed.")

# Training arguments
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--patience", type=int, default=10,
                    help="Waiting time for early stopping.")
parser.add_argument("--tolerance", type=float, default=0.05,
                    help="Tolerance value for the early stopping.")

# Pretraining arguments
parser.add_argument("-t", "--transfer_learning", default=None, type=str,
                    help="If a value is given, use autoencoder pretraining."
                         "If an existing path is given, a pretrained autoencoder is used."
                         "Else a new autoencoder is trained")
parser.add_argument("--selection", default="acc", choices=["loss", "acc"], type=str,
                    help="Allow to choose which model of the experiment is loaded.")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--features_learning_rate", "-f_lr", type=float, default=None,
                    help="Learning rate applied to the convolutional layers."
                         "If None all the layers have the same learning rate.")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--sampler', '-s', default="random", type=str, choices=['random', 'weighted'],
                    help="Sampler choice (random, or weighted for imbalanced datasets)")

parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')
parser.add_argument('--num_threads', type=int, default=0,
                    help='Number of threads used.')


def main(options):

    # Check if model is implemented
    from utils import model
    import inspect
    import sys

    choices = []
    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError('The model wanted %s has not been implemented in the module subject_level.py' % options.model)

    if "mni" in options.preprocessing:
        options.preprocessing = "mni"
        print(options.preprocessing)

    torch.set_num_threads(options.num_threads)
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
                                        options.split, options.n_splits, options.baseline, options.preprocessing)

    data_train = MRIDataset(options.input_dir, training_tsv, options.preprocessing, transform=transformations)
    data_valid = MRIDataset(options.input_dir, valid_tsv, options.preprocessing, transform=transformations)

    train_sampler = generate_sampler(data_train, options.sampler)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=options.batch_size,
                              sampler=train_sampler,
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

    # Initialize the model
    print('Initialization of the model')
    model = create_model(options)

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if options.features_learning_rate is None:
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate)
    else:
        optimizer = eval("torch.optim." + options.optimizer)([
            {'params': filter(lambda x: x.requires_grad, model.features.parameters()),
             'lr': options.features_learning_rate},
            {'params': filter(lambda x: x.requires_grad, model.classifier.parameters())}
            ], lr=options.learning_rate)

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
