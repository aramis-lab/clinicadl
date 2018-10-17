import argparse

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import Hosseini

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D CNN")

parser.add_argument("-id", "--input_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/CAPS_ADNI_T1_SPM',
                           help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("-dt", "--diagnosis_tsv", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/CN_vs_AD_diagnosis.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-ld", "--log_dir", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch',
                           help="Path to log dir for tensorboard usage.")
parser.add_argument("-t", "--transfer_learning", default=False,
                           help="If do transfer learning")
parser.add_argument("--n_splits", default=5,
                    help="How many folds for the k-fold cross validation procedure.")
parser.add_argument("--shuffle", default=True,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--use_gpu", default=False, nargs='+', type=int,
                    help="If use gpu or cpu. Empty implies cpu usage.")

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")


# feel free to add more arguments as you need


def main(options):

    check_and_clean(options.log_dir)
    test_accuracy = np.zeros((options.n_splits,))

    transformations = transforms.Compose([ToTensor()])
    # Split on subject level
    split_subjects_to_tsv(options.diagnosis_tsv, n_splits=options.n_splits)

    for fi in range(options.n_splits):
        # Get the data.
        print("Running for the %d fold" % fi)

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
        use_cuda = options.use_gpu

        # Initial the model
        model = Hosseini()

        if use_cuda:
            model.cuda()
        else:
            model.cpu()

        # Binary cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()

        lr = options.learning_rate
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)

        best_valid_accuracy = 0.0
        best_epoch = 0

        writer_train = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "valid")))
        writer_test = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "test")))

        for epoch in range(options.epochs):
            print("At %d -th epoch." % epoch)
            imgs = train(model, train_loader, use_cuda, criterion, optimizer, writer_train, epoch)

            # at then end of each epoch, we validate one time for the model with the validation data
            acc_mean_valid = test(model, valid_loader, use_cuda, criterion, writer_valid, epoch)
            print("Scan level validation accuracy is %f at the end of epoch %d" % acc_mean_valid, epoch)

            is_best = acc_mean_valid > best_valid_accuracy
            if is_best:
                best_valid_accuracy = acc_mean_valid
                best_epoch = epoch
            save_checkpoint(model.state_dict(), is_best, os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi)))

        # using test data to get the final performance on the best model
        best_model = load_best(model, os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi)))
        acc_mean_test_subject = test(best_model, test_loader, use_cuda, criterion, writer_test, best_epoch)
        print("Subject level mean test accuracy for fold %d is: %f") % (fi, acc_mean_test_subject)
        test_accuracy[fi] = acc_mean_test_subject

        # save the graph and image
        writer_train.add_graph(model, imgs)

    print("\n\n")
    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print('Mean accuracy of testing set: %f' % np.mean(test_accuracy))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
