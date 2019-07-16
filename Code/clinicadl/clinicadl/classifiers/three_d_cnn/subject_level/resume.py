import argparse
from os import path
from time import time
import torch
from torch.utils.data import DataLoader

from utils.classification_utils import train, test, load_model, read_json
from utils.data_utils import MRIDataset, MinMaxNormalization, load_data
from utils.model import create_model

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("model_path", type=str,
                    help="model selected")
parser.add_argument("split", type=int,
                    help="Will load the specific split wanted.")

# Computational argument
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--num_threads', type=int, default=0,
                    help='Number of threads used.')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    options = read_json(options, "CNN")
    print(path.exists(options.model_path))

    # Check if model is implemented
    from utils import model
    import inspect

    choices = []
    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError('The model wanted %s has not been implemented in the module model.py' % options.model)

    torch.set_num_threads(options.num_threads)
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
                              shuffle=options.shuffle,
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
    model_dir = path.join(options.model_path, "best_model_dir", "CNN", "fold_" + str(options.split))
    model, current_epoch = load_model(model, model_dir, options.gpu, 'checkpoint.pth.tar')

    options.beginning_epoch = current_epoch + 1

    # Define criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_path = path.join(options.model_path, 'optimizer.pth.tar')
    if path.exists(optimizer_path):
        print('Loading optimizer')
        optimizer_dict = torch.load(optimizer_path)
        name = optimizer_dict["name"]
        optimizer = eval("torch.optim." + name)(filter(lambda x: x.requires_grad, model.parameters()))
        optimizer.load_state_dict(optimizer_dict["optimizer"])
    else:
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate)

    print('Resuming the training task')
    training_time = time()

    train(model, train_loader, valid_loader, criterion, optimizer, True, options)
    training_time = time() - training_time

    # Load best model
    best_model_dir = path.join(options.output_dir, 'best_model_dir', 'CNN', 'fold_' + str(options.split))
    best_model, best_epoch = load_model(model, path.join(best_model_dir, 'best_loss'), options.gpu)

    # Get best performance
    acc_mean_train_subject, _ = test(best_model, train_loader, options.gpu, criterion)
    acc_mean_valid_subject, _ = test(best_model, valid_loader, options.gpu, criterion)
    log_dir = path.join(options.output_dir, 'log_dir', 'CNN', 'fold_' + str(options.split))

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
    text_file = open(path.join(log_dir, 'fold_output.txt'), 'w')
    text_file.write('Loss selection \n')
    text_file.write('Best loss : %i \n' % best_epoch)
    text_file.write('Time of training: %d s \n' % total_time)
    text_file.write('Training accuracy: %.2f %% \n' % (acc_mean_train_subject * 100))
    text_file.write('Validation accuracy: %.2f %% \n' % (acc_mean_valid_subject * 100))
    text_file.close()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
