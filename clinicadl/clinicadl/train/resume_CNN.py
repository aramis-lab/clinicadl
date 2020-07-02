# coding: utf8

import argparse
from os import path
from time import time
import torch
from torch.utils.data import DataLoader

from clinicadl.test.evaluation_singleCNN import test_cnn
from clinicadl.tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data
from clinicadl.tools.deep_learning import create_model, load_model, load_optimizer, read_json
from clinicadl.tools.deep_learning.cnn_utils import train

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="model selected")
parser.add_argument("split", type=int,
                    help="Will load the specific split wanted.")

# Computational argument
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    options = read_json(options)

    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()

    # Get the data.
    training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                        options.split, options.n_splits, options.baseline)

    data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations, preprocessing=options.preprocessing)
    data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations, preprocessing=options.preprocessing)

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
    model = create_model(options.model, options.gpu, dropout=options.dropout)
    model_dir = path.join(options.model_path, "best_model_dir", "CNN", "fold_" + str(options.split))
    model, current_epoch = load_model(model, model_dir, options.gpu, 'checkpoint.pth.tar')

    options.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_path = path.join(options.model_path, 'optimizer.pth.tar')
    optimizer = load_optimizer(optimizer_path, model)

    # Define output directories
    log_dir = path.join(options.output_dir, 'log_dir', 'fold_%i' % options.split, 'CNN')
    model_dir = path.join(options.output_dir, 'best_model_dir', 'fold_%i' % options.split, 'CNN')

    print('Resuming the training task')
    train(model, train_loader, valid_loader, criterion, optimizer, True, log_dir, model_dir, options)

    options.model_path = options.output_dir
    test_cnn(train_loader, "train", options.split, criterion, options)
    test_cnn(valid_loader, "validation", options.split, criterion, options)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
