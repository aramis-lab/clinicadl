import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from classification_utils import *
from model import *
import torchvision.transforms as transforms
import time
from torchvision.utils import save_image

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D autoencoder")

parser.add_argument("-id", "--caps_directory", default='/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("-dt", "--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/train/AD_vs_CN_baseline.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-od", "--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_ae_conv',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("-dty", "--data_type", default="from_MRI", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
parser.add_argument("--network", default="StackedConvDenAutoencoder", choices=["StackedConvDenAutoencoder"],
                    help="Autoencoder network type. (default=SparseAutoencoder)")

parser.add_argument("--patch_size", default="64", type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default="64", type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--batch_size", default=16, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')

# Training arguments
parser.add_argument("--epochs", default=100, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--use_gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


def main(options):

    if not os.path.exists(options.output_dir):
        os.makedirs(options.output_dir)
    check_and_clean(options.output_dir)

    ## Train the model with autoencoder

    print('Start the training for convolutional autoencoder and testing with a linear classifier, the weight and bias of each ConvAE will be saved for future use!')
    try:
        model = eval(options.network)()
    except:
        raise Exception('The model has not been implemented')

    ## need to normalized the value to [0, 1]
    transformations = transforms.Compose([CustomNormalizeMinMax()])

    training_tsv, valid_tsv = load_split(options.diagnosis_tsv)

    data_train = MRIDataset_patch(options.caps_directory, training_tsv, options.patch_size, options.patch_stride, transformations=transformations,
                                  data_type=options.data_type)
    data_valid = MRIDataset_patch(options.caps_directory, valid_tsv, options.patch_size, options.patch_stride, transformations=transformations,
                                  data_type=options.data_type)

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

    ## Decide to use gpu or cpu to train the autoencoder
    if options.use_gpu == False:
        use_cuda = False
        model.cpu()
        ## example image for tensorbordX usage:$
        example_batch = (next(iter(train_loader))['image'])[0, ...].unsqueeze(0)
    else:
        print("Using GPU")
        use_cuda = True
        model.cuda()
        ## example image for tensorbordX usage:$
        example_batch = (next(iter(train_loader))['image'].cuda())[0, ...].unsqueeze(0)

    writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "ConvDenAutoencoder", "train")))

    best_loss_eval = np.inf
    for epoch in range(options.epochs):
        if epoch % 10 == 0:
            if use_cuda:
                # Test the quality of our features with a randomly initialzed linear classifier.
                classifier = nn.Linear(512 * options.patch_size/(4*4*4), 2).cuda()
            else:
                classifier = nn.Linear(512 * options.patch_size/(4*4*4), 2)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=options.learning_rate)

        model.train()
        total_time = time.time()
        correct = 0
        for i, batch_data in enumerate(train_loader):

            if use_cuda:
                img, target = batch_data['image'].cuda(), batch_data['label'].cuda()
            else:
                img, target = batch_data['image'], batch_data['label']

            features = model(img).detach()
            prediction = classifier(features.view(features.size(0), -1))
            loss = criterion(prediction, target)

            ## save loss into tensorboardX
            writer_train.add_scalar('loss', loss, i + epoch * len(train_loader))
            print("For iteration %d, classification training loss is : %f" % (i + epoch * len(train_loader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        total_time = time.time() - total_time

        ## get the AE reconstruction loss and linear classifier training performances
        model.eval()
        features, x_reconstructed = model(img)
        reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)

        ## save the same slice from the reconstructed patch and original patch to do visual check
        if epoch % 10 == 0:
            print("Saving epoch {}".format(epoch))
            if not os.path.exists(os.path.join(options.output_dir, 'imgs')):
                os.makedirs(os.path.join(options.output_dir, 'imgs'))
            orig = extract_slice_img(img.cpu().data)
            save_image(orig, os.path.join(options.output_dir, 'imgs', 'orig_{}.png'.format(epoch)))
            pic = extract_slice_img(x_reconstructed.cpu().data)
            save_image(pic, os.path.join(options.output_dir, 'imgs', 'reconstruction_{}.png'.format(epoch)))

        print("For epoch %d, reconstruction loss is : %f" % (epoch, reconstruction_loss.item()))
        print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
        print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
            torch.mean(features.data), torch.max(features.data), torch.sum(features.data > 0.0)*100 / features.data.numel())
        )
        print("Linear classifier performance: {:.4f}%".format(correct.item() / float((len(train_loader)*options.batch_size))))
        print("="*80)

        # save the best model at the last epoch
        is_best = reconstruction_loss < best_loss_eval
        # is_best = True
        best_loss_eval = min(reconstruction_loss, best_loss_eval)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss_eval,
            'optimizer': optimizer.state_dict()
        }, is_best, os.path.join(options.output_dir, "best_model_dir"))

    ## reconstruct one example with the latest model
    visualize_ae(model, example_batch, path.join(options.output_dir, "imgs"))


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)