# coding: utf8

import argparse
import torch
from os import path
from time import time
from torch.utils.data import DataLoader

from .utils import ae_finetuning
from ..tools.deep_learning.data import MRIDataset, MinMaxNormalization, load_data
from ..tools.deep_learning import load_model, create_autoencoder, load_optimizer, read_json

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="model selected")
parser.add_argument("split", type=int,
                    help="Will load the specific split wanted.")

# Optimizer arguments
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    options = read_json(options, "ConvAutoencoder")

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

    data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations)
    data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations)

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
    decoder = create_autoencoder(options.model)

    decoder, current_epoch = load_model(decoder, options.model_path, options.gpu, 'checkpoint.pth.tar')
    if options.gpu:
        decoder = decoder.cuda()

    options.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer_path = path.join(options.model_path, 'optimizer.pth.tar')
    optimizer = load_optimizer(optimizer_path, decoder)

    # Define output directories
    log_dir = path.join(options.output_dir, 'log_dir', 'fold_%i' % options.split, 'ConvAutoencoder')
    visualization_dir = path.join(options.output_dir, 'visualize', 'fold_%i' % options.split)
    model_dir = path.join(options.output_dir, 'best_model_dir', 'fold_%i' % options.split, 'ConvAutoencoder')

    print('Resuming the training task')
    ae_finetuning(decoder, train_loader, valid_loader, criterion, optimizer, True,
                  log_dir, model_dir, visualization_dir, options)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
    text_file = open(path.join(options.log_dir, 'model_output.txt'), 'w')
    text_file.write('Time of training: %d s \n' % total_time)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
