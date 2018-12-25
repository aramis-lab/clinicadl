import argparse

from time import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from classification_utils import *
from data_utils import *
from model import *
import copy

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch autoencoder")

parser.add_argument("-id", "--caps_directory", default='/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("-dt", "--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-od", "--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_test',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--network", default="Autoencoder", choices=["Autoencoder", "ConvAutoencoder"],
                    help="Autoencoder network type. (default=VoxResNet)")
parser.add_argument("--patch_size", default="21", type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default="10", type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--num_workers", '-w', default=0, type=int,
                    help='the number of batch being loaded in parallel')
# Training arguments
parser.add_argument("--epochs", default=3, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# parser.add_argument("--test_sessions", default=["ses-M00"], nargs='+', type=str,
#                     help="Test the accuracy at the end of the model for the sessions selected")
# parser.add_argument("--visualization", action='store_true', default=False,
#                     help='Chooses if visualization is done on AE pretraining')
# parser.add_argument("-m", "--model", default='Conv_3', type=str, choices=["Conv_3", "Conv_4", "Test", "Test_nobatch", "Rieke", "Test2", 'Optim'],
#                     help="model selected")

def main(options):

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    check_and_clean(options.output_dir)

    ## Train the model with autoencoder

    print('Start the autoencoder!')
    try:
        autoencoder = eval(options.network)()
    except:
        raise Exception('The model has not been implemented')

    data_train = MRIDataset(options.caps_directory, options.diagnosis_tsv, options.patch_size, options.patch_stride)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=options.batch_size,
                              shuffle=options.shuffle,
                              num_workers=options.num_workers,
                              drop_last=True
                              )

    ## Decide to use gpu or cpu to train the autoencoder
    if options.use_gpu == False:
        use_cuda = False
        autoencoder.cpu()
    else:
        print("Using GPU")
        use_cuda = True
        autoencoder.cuda()

    # Define loss and optimizer, for autoencoder, better using MSELoss
    loss = torch.nn.MSELoss()
    lr = options.learning_rate
    # chosen optimer for back-propogation
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, autoencoder.parameters()), lr,
                                                         weight_decay=options.weight_decay)
    # apply exponential decay for learning rate
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    print('Beginning the training for autoencoder')
    # parameters used in training
    best_loss = 1e10
    writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "autoencoder_train")))

    for epoch_i in range(options.epochs):
        print("At %s -th epoch." % str(epoch_i))
        # train the ae
        example_imgs, epoch_loss = train_ae(autoencoder, train_loader, use_cuda, loss, optimizer, writer_train, epoch_i, options)

        ## update the learing rate
        if epoch_i % 20 == 0:
            scheduler.step()

        # save the best model based on the updated batch_loss for each epoch
        is_best = epoch_loss < best_loss
        best_loss = min(epoch_loss, best_loss)
        save_checkpoint({
            'epoch': epoch_i + 1,
            'state_dict': autoencoder.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        }, is_best, os.path.join(options.output_dir, "best_model_dir"))

    ## save the graph and image
    writer_train.add_graph(autoencoder, example_imgs)

if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
