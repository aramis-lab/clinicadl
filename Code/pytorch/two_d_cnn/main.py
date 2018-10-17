import argparse

import numpy as np
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import cuda
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import alexnet2D

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D CNN")

parser.add_argument("-id", "--input_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/BIDS/ADNI_BIDS_T1_new',
                           help="Path to input dir of the MRI, it could be BIDS_dir or preprocessed CAPS_dir.")
parser.add_argument("-dt", "--diagnosis_tsv", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/CN_vs_AD_diagnosis.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-ld", "--log_dir", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/Results/pytorch',
                           help="Path to log dir for tensorboard usage.")
parser.add_argument("-t", "--transfer_learning", default=True,
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

    trg_size = (224, 224) ## this is the original input size of alexnet
    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])
    # Split on subject level
    split_subjects_to_tsv(options.diagnosis_tsv, n_splits=options.n_splits)

    for fi in range(options.n_splits):
        # Get the data.
        print("Running for the %d fold" % fi)

        training_tsv, test_tsv, valid_tsv = load_split(options.diagnosis_tsv, fold=fi)


        data_train = AD_Standard_2DSlicesData(options.input_dir, training_tsv, transformations)
        data_test = AD_Standard_2DSlicesData(options.input_dir, test_tsv, transformations)
        data_valid = AD_Standard_2DSlicesData(options.input_dir, valid_tsv, transformations)

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
                                 drop_last=True
                                 )

        valid_loader = DataLoader(data_valid,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True
                                 )

        if not options.use_gpu:
            use_cuda = options.use_gpu
        else:
            use_cuda = True

        # Initial the model
        model = alexnet2D(pretrained=options.transfer_learning)

        if use_cuda:
            model.cuda()
        else:
            model.cpu()

        # Binary cross-entropy loss
        criterion = torch.nn.CrossEntropyLoss()

        lr = options.learning_rate
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)

        best_accuracy = 0.0

        writer_train = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "valid")))
        writer_test = SummaryWriter(log_dir=(os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi), "test")))

        for epoch_i in range(options.epochs):
            print("At %d -th epoch." % epoch_i)
            imgs = train(model, train_loader, use_cuda, criterion, optimizer, writer_train, epoch_i)

            ## at then end of each epoch, we validate one time for the model with the validation data
            acc_mean_valid = validate(model, valid_loader, use_cuda, criterion, writer_valid, epoch_i)
            print("Slice level average validation accuracy is %f at the end of epoch %d") % (acc_mean_valid, epoch_i)

            is_best = acc_mean_valid > best_accuracy
            best_prec1 = max(best_accuracy, acc_mean_valid)
            save_checkpoint({
                'epoch': epoch_i + 1,
                'state_dict': model.state_dict(),
                'best_predic1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, os.path.join(options.log_dir, "log_dir" + "_fold" + str(fi)))

        ### using test data to get the final performance
        acc_mean_test_subject = test(model, test_loader, use_cuda, writer_test)
        print("Subject level mean test accuracy for fold %d is: %f") % (fi, acc_mean_test_subject)
        test_accuracy[fi] = acc_mean_test_subject

        ## save the graph and image
        writer_train.add_graph(model, imgs)

    print("\n\n")
    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print('\nMean accuray of testing set: %f') % (np.mean(test_accuracy))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s") % (parser.parse_known_args()[1])
    main(options)
