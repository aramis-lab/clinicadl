import argparse

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import Hosseini

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_tsv", type=str,
                    help="Path to tsv file of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("log_dir", type=str,
                    help="Path to log dir for tensorboard usage.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")

# Optional arguments
parser.add_argument("-t", "--transfer_learning", default=False,
                    help="If do transfer learning")
parser.add_argument("--n_splits", default=5, type=int,
                    help="How many folds for the k-fold cross validation procedure.")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument('--accumulation_steps', '-asteps', default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')


# feel free to add more arguments as you need


def main(options):

    check_and_clean(options.log_dir)
    test_accuracy = np.zeros(options.n_splits)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    transformations = transforms.Compose([ToTensor()])
    # Split on subject level
    split_subjects_to_tsv(options.diagnosis_tsv, n_splits=options.n_splits)

    for fi in range(options.n_splits):
        # Get the data.
        print("Running for fold %d" % fi)

        training_tsv, test_tsv, valid_tsv = load_split(options.diagnosis_tsv, fold=fi)

        data_train = MRIDataset(options.input_dir, training_tsv, transformations)
        data_test = MRIDataset(options.input_dir, test_tsv, transformations)
        data_valid = MRIDataset(options.input_dir, valid_tsv, transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=0,
                                  drop_last=True
                                  )

        test_loader = DataLoader(data_test,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False
                                 )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False
                                  )

        # Initialize the model
        model = Hosseini()
        if options.gpu:
            model.cuda()
        else:
            model.cpu()

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate)

        train(model, train_loader, valid_loader, criterion, optimizer, fi, options)

        # Load best model
        best_model, best_epoch = load_best(model, os.path.join(options.log_dir, "fold" + str(fi)))

        # Get test performance
        acc_mean_test_subject = test(best_model, test_loader, options.gpu)
        print("Subject level mean test accuracy for fold %d is: %f" % (fi, acc_mean_test_subject))
        test_accuracy[fi] = acc_mean_test_subject
        print()

    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print('Mean accuracy of testing set: %f' % np.mean(test_accuracy))
    test_df = pd.DataFrame(test_accuracy, columns=['accuracy'])
    test_df.index.name = 'fold'
    test_df.to_csv(path.join(options.log_dir, 'test_accuracies.tsv'), sep='\t')


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
