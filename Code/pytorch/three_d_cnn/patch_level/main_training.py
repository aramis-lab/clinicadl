import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from classification_utils import *
import copy
from model import *

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch CNN. The input MRI's dimension is 169*208*179 after cropping")

parser.add_argument("-id", "--caps_directory", default='/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("-dt", "--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-od", "--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_test',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("-nw", "--network", default="AllConvNet3D", choices=["VoxResNet", "AllConvNet3D"],
                    help="Deep network type. (default=VoxResNet)")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("-dty", "--data_type", default="from_patch", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")

parser.add_argument("--patch_size", default="21", type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default="21", type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--runs", default=1, type=int,
                    help="Number of runs with the same training / validation split.")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')

# transfer learning
parser.add_argument("-tla", "--transfer_learning_autoencoder", default=False, action='store_true',
                    help="If do transfer learning using autoencoder, the learnt weights will be transferred")
parser.add_argument("-tlt", "--transfer_learning_task", default=False, action='store_true',
                    help="If do transfer learning using different tasks, the learnt weights will be transferred")
parser.add_argument("-tbm", "--transfer_learnt_best_model", default=False, action='store_true',
                    help="The path to save the transfer learning model")

# Training arguments
parser.add_argument("--epochs", default=3, type=int,
                    help="Epochs through the data. (default=20)")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--random_state', default=None,
                    help='If set random state when splitting data training and validation set using StratifiedShuffleSplit')

def main(options):

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    check_and_clean(options.output_dir)

    ## Train the model with transfer learning
    if options.transfer_learning_autoencoder:
        print('Train the model with the weights from a pre-trained model by autoencoder!')
        print('The chosen network is %s !' % options.network)

        try:
            model = eval(options.network)()
        except:
            raise Exception('The model has not been implemented')

        pretraining_model = torch.load(options.transfer_learnt_best_model)

        ## convert the weight and bias into the current model
        model.state_dict()['conv1.weight'] = pretraining_model['encoder.weight']
        model.state_dict()['conv1.bias'] = pretraining_model['encoder.bias']

    elif options.transfer_learning_task:
        print('Train the model with the weights from a pre-trained model by different tasks!')
        print('The chosen network is %s !' % options.network)
        model = torch.load(options.transfer_learnt_best_model)

    else:
        print('Train the model from scratch!')
        print('The chosen network is %s !' % options.network)
        try:
            model = eval(options.network)()
        except:
            raise Exception('The model has not been implemented')

    ## the inital model weight and bias
    init_state = copy.deepcopy(model.state_dict())

    for fi in range(options.runs):
        print("Running for the %d run" % fi)
        model.load_state_dict(init_state)

        training_tsv, valid_tsv = load_split(options.diagnosis_tsv, random_state=options.random_state)
        data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size, options.patch_stride, data_type=options.data_type)
        data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size, options.patch_stride, data_type=options.data_type)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=options.num_workers,
                                  drop_last=True,
                                  pin_memory=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False,
                                  pin_memory=True
                                  )
        ## Decide to use gpu or cpu to train the model
        if options.use_gpu == False:
            use_cuda = False
            model.cpu()
            example_batch = next(iter(train_loader))['image']

        else:
            print("Using GPU")
            use_cuda = True
            model.cuda()
            ## example image for tensorbordX usage:$
            example_batch = next(iter(train_loader))['image'].cuda()


        # Define loss and optimizer
        loss = torch.nn.CrossEntropyLoss()

        lr = options.learning_rate
        # chosen optimer for back-propogation
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr,
                                                             weight_decay=options.weight_decay)
        # apply exponential decay for learning rate
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        print('Beginning the training task')
        # parameters used in training
        best_accuracy = 0.0
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "iteration_" + str(fi), "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "iteration_" + str(fi), "valid")))

        ## get the info for training and write them into tsv files.
        train_subjects = []
        valid_subjects = []
        y_grounds_train = []
        y_grounds_valid = []
        y_hats_train = []
        y_hats_valid = []

        for epoch_i in range(options.epochs):
            print("At %s -th epoch." % str(epoch_i))

            # train the model
            train_subject, y_ground_train, y_hat_train, acc_mean_train, global_steps_train = train(model, train_loader, use_cuda, loss, optimizer, writer_train, epoch_i, model_mode='train')
            train_subjects.extend(train_subject)
            y_grounds_train.extend(y_ground_train)
            y_hats_train.extend(y_hat_train)
            ## at then end of each epoch, we validate one time for the model with the validation data
            _, valid_subject, y_ground_valid, y_hat_valid, acc_mean_valid, global_steps_valid = train(model, valid_loader, use_cuda, loss, optimizer, writer_valid, epoch_i, model_mode='valid', global_steps=global_steps_train)
            print("Slice level average validation accuracy is %f at the end of epoch %d" % (acc_mean_valid, epoch_i))
            valid_subjects.extend(valid_subject)
            y_grounds_valid.extend(y_ground_valid)
            y_hats_valid.extend(y_hat_valid)

            ## update the learing rate
            if epoch_i % 20 == 0:
                scheduler.step()

            # save the best model on the validation dataset
            is_best = acc_mean_valid > best_accuracy
            best_accuracy = max(best_accuracy, acc_mean_valid)
            save_checkpoint({
                'epoch': epoch_i + 1,
                'state_dict': model.state_dict(),
                'best_predict': best_accuracy,
                'optimizer': optimizer.state_dict()
            }, is_best, os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi)))

        # ### using test data to get the final performance
        # ## take the best_validated model for test
        # if os.path.isfile(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")):
        #     print("=> loading checkpoint '{}'".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")))
        #     checkpoint = torch.load(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar"))
        #     best_epoch = checkpoint['epoch']
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     print("=> loaded model '{}' for the best perfomrmance at (epoch {})".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar"), best_epoch))
        # else:
        #     print("=> no checkpoint found at '{}'".format(os.path.join(options.output_dir, "best_model_dir", "iteration_" + str(fi), "model_best.pth.tar")))
        #
        # imgs_test, test_subject, y_ground_test, y_hat_test, acc_mean_test, global_steps_test = train(model, test_loader, use_cuda, loss, optimizer, writer_test, 0, model_mode='test')
        # test_subjects.extend(test_subject)
        # y_grounds_test.extend(y_ground_test)
        # y_hats_test.extend(y_hat_test)
        # print("Slice level mean test accuracy for fold %d is: %f" % (fi, acc_mean_test))
        # test_accuracy[fi] = acc_mean_test

        ## save the graph and image
        writer_train.add_graph(model, example_batch)

        ### write the information of subjects and performances into tsv files.
        iteration_subjects_df_train, results_train = results_to_tsvs(options.output_dir, fi, train_subjects, y_grounds_train, y_hats_train, mode='train')
        iteration_subjects_df_valid, results_valid = results_to_tsvs(options.output_dir, fi, valid_subjects, y_grounds_valid, y_hats_valid, mode='validation')


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
