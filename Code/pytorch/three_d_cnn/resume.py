import argparse
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import *

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to tsv files of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("model_path", type=str,
                    help="model selected")

# Default values not specified in previous models
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization for visualization")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--n_splits", type=int, default=None,
                    help="If a value is given will load data of a k-fold CV")
parser.add_argument("--split", type=int, default=0,
                    help="Will load the specific split wanted.")


# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def correct_model_options(model_options):
    corrected_options = []
    for i, option in enumerate(model_options):
        if '-' not in option:
            new_option = '_'.join([corrected_options[-1], option])
            corrected_options[-1] = new_option
        else:
            corrected_options.append(option)

    return corrected_options


def parse_model_name(model_path, options, position=-1):
    model_name = model_path.split(os.sep)[position]
    model_options = model_name.split('_')
    model_options = correct_model_options(model_options)
    options.log_dir = os.path.abspath(os.path.join(options.model_path, os.pardir))

    for option in model_options:
        option_split = option.split("-")
        key = option_split[0]
        if len(option_split) > 2:
            content = "-".join(option_split[1:])
        else:
            content = option_split[1]

        if key == 'model':
            options.model = content
        elif key == 'task':
            diagnoses = content.split('_')
            if 'baseline' in diagnoses:
                options.baseline = True
                diagnoses.remove('baseline')
            else:
                options.baseline = False
            if options.diagnoses is None:
                options.diagnoses = diagnoses
        elif key == 'gpu':
            options.gpu = bool(content)
        elif key == 'epochs':
            options.epochs = int(content)
        elif key == 'workers':
            options.num_workers = int(content)
        elif key == 'threads':
            options.num_threads = int(content)
        elif key == 'lr':
            options.learning_rate = float(content)
        elif key == 'norm':
            options.minmaxnormalization = bool(content)
        elif key == 'batch':
            options.batch_size = int(content)
        elif key == 'acc':
            options.accumulation_steps = int(content)
        elif key == 'eval':
            options.evaluation_steps = int(content)
        elif key == 'splits':
            options.n_splits = int(content)
        elif key == 'split':
            options.split = int(content)
        elif key == 'preprocessing':
            options.preprocessing = content

    return options


def main(options):

    options = parse_model_name(options.model_path, options)
    print(path.exists(options.model_path))

    # Check if model is implemented
    import model
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
    model = eval(options.model)()
    model, current_epoch = load_model(model, options.model_path, 'checkpoint.pth.tar')
    if options.gpu:
        model = model.cuda()

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
    # TODO deal with the run parameter (str 'runX') and modify train accordingly

    train(model, train_loader, valid_loader, criterion, optimizer, True, options)
    training_time = time() - training_time

    # Load best model
    best_model, best_epoch = load_model(model, os.path.join(options.model_path))

    # Get best performance
    acc_mean_train_subject, _ = test(best_model, train_loader, options.gpu, criterion)
    acc_mean_valid_subject, _ = test(best_model, valid_loader, options.gpu, criterion)
    accuracies = (acc_mean_train_subject, acc_mean_valid_subject)
    write_summary(options.log_dir, run, accuracies, best_epoch, training_time)

    del best_model

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
    text_file = open(path.join(options.log_dir, 'model_output.txt'), 'w')
    text_file.write('Time of training: %d s \n' % total_time)


def write_summary(log_dir, run, accuracies, best_epoch, time):
    fold_dir = path.join(log_dir, "run" + str(run))
    text_file = open(path.join(fold_dir, 'run_output.txt'), 'w')
    text_file.write('Fold: %i \n' % run)
    text_file.write('Best epoch: %i \n' % best_epoch)
    text_file.write('Time of training: %d s \n' % time)
    text_file.write('Accuracy on training set: %.2f %% \n' % accuracies[0])
    text_file.write('Accuracy on validation set: %.2f %% \n' % accuracies[1])
    text_file.close()


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline)
    options = commandline[0]
    if commandline[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
