import argparse
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from classification_utils import *
from model import alexnet2D

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D CNN")

parser.add_argument("-id", "--caps_directory", default='/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("-dt", "--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-od", "--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
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
parser.add_argument("--use_gpu", default=False, nargs='+',
                    help="If use gpu or cpu. Empty implies cpu usage.")
parser.add_argument('--force', default=True,
                    help='If force to rerun the classification, default behavior is to clean the output folder and restart from scratch')

# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
# parser.add_argument("--estop", default=1e-2, type=float,
#                     help="Early stopping criteria on the development set. (default=1e-2)")

def main(options):

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)

    if options.force == True:
        check_and_clean(options.output_dir)

    test_accuracy = np.zeros((options.n_splits,))
    trg_size = (224, 224) ## this is the original input size of alexnet
    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])

    # Split the data into 5 fold on subject-level
    split_subjects_to_tsv(options.diagnosis_tsv, n_splits=options.n_splits)

    for fi in range(options.n_splits):
        # Get the data.
        print("Running for the %d iteration" % fi)

        ## load the tsv file
        training_tsv, test_tsv, valid_tsv = load_split(options.diagnosis_tsv, fi, options.n_splits, val_size=0.15)

        data_train = mri_to_rgb_transfer(options.caps_directory, training_tsv, transformations)
        data_test = mri_to_rgb_transfer(options.caps_directory, test_tsv, transformations)
        data_valid = mri_to_rgb_transfer(options.caps_directory, valid_tsv, transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=0,
                                  drop_last=True)

        test_loader = DataLoader(data_test,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True)

        valid_loader = DataLoader(data_valid,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=True)


        # Initial the model
        model = alexnet2D(pretrained=options.transfer_learning)

        ## Decide to use gpu or cpu to train the model
        if options.use_gpu == False:
            use_cuda = False
            model.cpu()
        else:
            use_cuda = True
            model.cuda()

        # Binary cross-entropy loss
        loss = torch.nn.CrossEntropyLoss()
        # learning rate for training
        lr = options.learning_rate
        # chosen optimer for back-propogation
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)

        # parameters used in training
        best_accuracy = 0.0
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "iteration_" + str(fi), "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "iteration_" + str(fi), "valid")))
        writer_test = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "iteration_" + str(fi), "test")))

        ## get the info for training and write them into tsv files.
        train_subjects = []
        valid_subjects = []
        test_subjects = []
        y_grounds_train = []
        y_grounds_valid = []
        y_grounds_test = []
        y_hats_train = []
        y_hats_valid = []
        y_hats_test = []

        for epoch_i in range(options.epochs):
            print("At %d -th epoch.") % (epoch_i)

            # train the model
            imgs_train, train_subject, y_ground_train, y_hat_train, acc_mean_train = train(model, train_loader, use_cuda, loss, optimizer, writer_train, epoch_i, train_mode='train')
            train_subjects.extend(train_subject)
            y_grounds_train.extend(y_ground_train)
            y_hats_train.extend(y_hat_train)
            ## at then end of each epoch, we validate one time for the model with the validation data
            imgs_valid, valid_subject, y_ground_valid, y_hat_valid, acc_mean_valid = train(model, valid_loader, use_cuda, loss, optimizer, writer_valid, epoch_i, train_mode='valid')
            print("Slice level average validation accuracy is %f at the end of epoch %d") % (acc_mean_valid, epoch_i)
            valid_subjects.extend(valid_subject)
            y_grounds_valid.extend(y_ground_valid)
            y_hats_valid.extend(y_hat_valid)

            # save the best model on the validation dataset
            is_best = acc_mean_valid > best_accuracy
            best_prec1 = max(best_accuracy, acc_mean_valid)
            save_checkpoint({
                'epoch': epoch_i + 1,
                'state_dict': model.state_dict(),
                'best_predic1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi)))

        ### using test data to get the final performance
        ## take the best_validated model for test
        if os.path.isfile(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")):
            print("=> loading checkpoint '{}'".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")))
            checkpoint = torch.load(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar"))
            best_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded model '{}' for the best perfomrmance at (epoch {})".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar"), best_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")))

        imgs_test, test_subject, y_ground_test, y_hat_test, acc_mean_test = train(model, test_loader, use_cuda, loss, optimizer, writer_test, 0, train_mode='test')
        test_subjects.extend(test_subject)
        y_grounds_test.extend(y_ground_test)
        y_hats_test.extend(y_hat_test)
        print("Slice level mean test accuracy for fold %d is: %f") % (fi, acc_mean_test)
        test_accuracy[fi] = acc_mean_test

        ## save the graph and image
        writer_train.add_graph(model, imgs_train)

        ### write the information of subjects and performances into tsv files.
        iteration_subjects_df_train, results_train = results_to_tsvs(options.output_dir, fi, train_subjects, y_grounds_train, y_hats_train)
        iteration_subjects_df_valid, results_valid = results_to_tsvs(options.output_dir, fi, valid_subjects, y_grounds_valid, y_hats_valid)
        iteration_subjects_df_test, results_test = results_to_tsvs(options.output_dir, fi, test_subjects, y_grounds_test, y_hats_test)

    print("\n\n")
    print("For the k-fold CV, testing accuracies are %s " % str(test_accuracy))
    print('\nMean accuray of testing set: %f') % (np.mean(test_accuracy))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s") % (parser.parse_known_args()[1])
    main(options)
