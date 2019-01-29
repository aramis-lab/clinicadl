import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from classification_utils import *
import copy
from model import *
import torchvision.transforms as transforms

__author__ = "Junhao Wen, Elina Thibeausutre"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch CNN")

## Data arguments
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_ae_conv',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--data_type", default="from_patch", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
parser.add_argument("--patch_size", default=51, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=51, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--batch_size", default=8, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--random_state", default=544423, type=int,
                    help='If set random state when splitting data training and validation set using StratifiedShuffleSplit')
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')

# transfer learning
parser.add_argument("--network", default="Conv_4_FC_2", choices=["Conv_4_FC_2", "VoxResNet", "AllConvNet3D"],
                    help="Autoencoder network type. (default=Conv_4_FC_2). Also, you can try training from scratch using VoxResNet and AllConvNet3D")
parser.add_argument("--transfer_learning_autoencoder", default=True, type=bool,
                    help="If do transfer learning using autoencoder, the learnt weights will be transferred. Should be exclusive with net_work")
parser.add_argument("--train_from_stop_point", default=False, type=bool,
                    help='If train a network from the very beginning or from the point where it stopped, where the network is saved by tensorboardX')

# Training arguments
parser.add_argument("--epochs", default=1, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--runs", default=1, type=int,
                    help="Number of runs with the same training / validation split.")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--use_gpu', default=True, type=bool,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')

def main(options):

    if options.train_from_stop_point:
        ## only delete the CNN output, not the AE output
        check_and_clean(os.path.join(options.output_dir, 'best_model_dir', 'CNN'))
        check_and_clean(os.path.join(options.output_dir, 'log_dir', 'CNN'))
        check_and_clean(os.path.join(options.output_dir, 'performances'))
    else:
        print("Train the model from 0 epoch")

    ## Train the model with pretrained AE
    if options.transfer_learning_autoencoder:
        print('Train the model with the weights from a pre-trained model by autoencoder!')
        print('The chosen network is %s !' % options.network)
        print('The chosen network should correspond to the encoder of the stacked pretrained AE')

        try:
            model = eval(options.network)()
        except:
            raise Exception('The model has not been implemented or has bugs in the model codes')
        model, _ = load_model_after_ae(model, os.path.join(options.output_dir, 'log_dir', 'best_model_dir'), filename='model_pretrained_with_AE.pth.tar')
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

        ## need to normalized the value to [0, 1]
        transformations = transforms.Compose([NormalizeMinMax()])

        training_tsv, valid_tsv = load_split(options.diagnosis_tsv, random_state=options.random_state)
        data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size, options.patch_stride, transformations=transformations, data_type=options.data_type)
        data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size, options.patch_stride, transformations=transformations, data_type=options.data_type)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=options.num_workers,
                                  drop_last=True,
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False,
                                  )

        lr = options.learning_rate
        # chosen optimer for back-propogation
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr,
                                                             weight_decay=options.weight_decay)
        # apply exponential decay for learning rate
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        ## if train a model at the stopping point?
        if options.train_from_stop_point:
            model, optimizer, global_step, global_epoch = load_model_from_log(model, optimizer, os.path.join(options.output_dir, 'best_model_dir', 'CNN', "iteration_" + str(fi)),
                                           filename='checkpoint.pth.tar')
        else:
            global_step = 0

        model.load_state_dict(init_state)

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

        print('Beginning the training task')
        # parameters used in training
        best_accuracy = 0.0
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "CNN", "iteration_" + str(fi), "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "CNN", "iteration_" + str(fi), "valid")))

        ## get the info for training and write them into tsv files.
        train_subjects = []
        valid_subjects = []
        y_grounds_train = []
        y_grounds_valid = []
        y_hats_train = []
        y_hats_valid = []

        for epoch in range(options.epochs):
            
            if options.train_from_stop_point:
                epoch += global_epoch
                
            print("At %s -th epoch." % str(epoch))

            # train the model
            train_subject, y_ground_train, y_hat_train, acc_mean_train, global_step = train(model, train_loader, use_cuda, loss, optimizer, writer_train, epoch, model_mode='train', global_step=global_step)
            train_subjects.extend(train_subject)
            y_grounds_train.extend(y_ground_train)
            y_hats_train.extend(y_hat_train)
            ## at then end of each epoch, we validate one time for the model with the validation data
            valid_subject, y_ground_valid, y_hat_valid, acc_mean_valid, global_step = train(model, valid_loader, use_cuda, loss, optimizer, writer_valid, epoch, model_mode='valid', global_step=global_step)
            print("Slice level average validation accuracy is %f at the end of epoch %d" % (acc_mean_valid, epoch))
            valid_subjects.extend(valid_subject)
            y_grounds_valid.extend(y_ground_valid)
            y_hats_valid.extend(y_hat_valid)

            ## update the learing rate
            if epoch % 20 == 0:
                scheduler.step()

            # save the best model on the validation dataset
            is_best = acc_mean_valid > best_accuracy
            best_accuracy = max(best_accuracy, acc_mean_valid)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_predict': best_accuracy,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step
            }, is_best, os.path.join(options.output_dir, "best_model_dir", "CNN", "iteration_" + str(fi)))

        ## save the graph and image
        writer_train.add_graph(model, example_batch)

        ### write the information of subjects and performances into tsv files.
        iteration_subjects_df_train, results_train = results_to_tsvs(options.output_dir, fi, train_subjects, y_grounds_train, y_hats_train, mode='train')
        iteration_subjects_df_valid, results_valid = results_to_tsvs(options.output_dir, fi, valid_subjects, y_grounds_valid, y_hats_valid, mode='validation')


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    ## save the commind line arguments into a tsv file for tracing all different kinds of experiments
    commandline_to_jason(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
