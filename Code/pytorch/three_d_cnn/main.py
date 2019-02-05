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
parser.add_argument("log_dir", type=str,
                    help="Path to log dir for tensorboard usage.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("model", type=str,
                    help="model selected")

# Data Management
parser.add_argument("--data_path", default="linear", choices=["linear", "dartel", "mni"], type=str,
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
parser.add_argument("--runs", default=1, type=int,
                    help="Number of runs with the same training / validation split.")
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

# Pretraining arguments
parser.add_argument("-t", "--transfer_learning", default=None, type=str,
                    help="If a value is given, use autoencoder pretraining."
                         "If an existing path is given, a pretrained autoencoder is used."
                         "Else a new autoencoder is trained")
parser.add_argument("--transfer_learning_diagnoses", "-t_diagnoses", type=str, default=None, nargs='+',
                    help='If transfer learning, gives the diagnoses to use to perform pretraining')
parser.add_argument("--transfer_learning_epochs", "-t_e", type=int, default=10,
                    help="Number of epochs for pretraining")
parser.add_argument("--transfer_learning_rate", "-t_lr", type=float, default=1e-4,
                    help='The learning rate used for AE pretraining')
parser.add_argument("--features_learning_rate", "-f_lr", type=float, default=None,
                    help="Learning rate applied to the convolutional layers."
                         "If None all the layers have the same learning rate.")
parser.add_argument("--visualization", action='store_true', default=False,
                    help='Chooses if visualization is done on AE pretraining')
parser.add_argument("--transfer_difference", "-t_diff", type=int, default=0,
                    help="Difference of convolutional layers between current model and pretrained model")
parser.add_argument("--add_sigmoid", default=False, action="store_true",
                    help="Ad sigmoid function at the end of the decoder.")

# Training arguments
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--tolerance", "-tol", default=5e-2, type=float,
                    help="Allows to stop when the training data is nearly learnt")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')
parser.add_argument('--num_threads', type=int, default=1,
                    help='Number of threads used.')


def main(options):

    # Check if model is implemented
    import model
    import inspect

    choices = []
    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError('The model wanted %s has not been implemented in the module model.py' % options.model)

    check_and_clean(options.log_dir)
    torch.set_num_threads(options.num_threads)
    valid_accuracies = np.zeros(options.runs)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    total_time = time()
    # Pretraining the model
    if options.transfer_learning is not None:
        model = eval(options.model)()
        criterion = torch.nn.MSELoss()

        if path.exists(options.transfer_learning):
            print("A pretrained autoencoder is loaded at path %s" % options.transfer_learning)
            apply_autoencoder_weights(model, options.transfer_learning, options.log_dir, options.transfer_difference)

        else:
            if options.transfer_learning_diagnoses is None:
                raise Exception("Diagnosis labels must be given to train the autoencoder.")
            training_tsv, valid_tsv = load_data(options.diagnosis_path, options.transfer_learning_diagnoses,
                                                options.split, options.n_splits, options.baseline)

            data_train = MRIDataset(options.input_dir, training_tsv, options.data_path, transformations)
            data_valid = MRIDataset(options.input_dir, valid_tsv, options.data_path, transformations)

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

            pretraining_dir = path.join(options.log_dir, 'pretraining')
            greedy_learning(model, train_loader, valid_loader, criterion, True, pretraining_dir, options)

    for run in range(options.runs):
        # Get the data.
        training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                            options.split, options.n_splits, options.baseline)

        data_train = MRIDataset(options.input_dir, training_tsv, options.data_path, transform=transformations)
        data_valid = MRIDataset(options.input_dir, valid_tsv, options.data_path, transform=transformations)

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
        training_time = time()
        train(model, train_loader, valid_loader, criterion, optimizer, run, options)
        training_time = time() - training_time

        # Load best model
        best_model, best_epoch = load_model(model, os.path.join(options.log_dir, "run" + str(run)))

        # Get best performance
        acc_mean_train_subject, _ = test(best_model, train_loader, options.gpu, criterion)
        acc_mean_valid_subject, _ = test(best_model, valid_loader, options.gpu, criterion)
        valid_accuracies[run] = acc_mean_valid_subject
        accuracies = (acc_mean_train_subject, acc_mean_valid_subject)
        write_summary(options.log_dir, run, accuracies, best_epoch, training_time)

        del best_model

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
    text_file = open(path.join(options.log_dir, 'model_output.txt'), 'w')
    text_file.write('Time of training: %d s \n' % total_time)
    text_file.write('Mean best validation accuracy: %.2f %% \n' % np.mean(valid_accuracies))
    text_file.write('Standard variation of best validation accuracy: %.2f %% \n' % np.std(valid_accuracies))
    text_file.close()


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
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
