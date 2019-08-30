import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .utils import MRIDataset_patch_hippocampus, test, hard_voting_to_tsvs, soft_voting_to_tsvs
from tools.deep_learning.data import MinMaxNormalization
from tools.deep_learning import create_model, load_model


__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for test of hippocampus approach")

# Mandatory arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to tsv file of the population based on the diagnosis tsv files."
                         "To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs and the tsv files containing the performances.")

# Test parameters
parser.add_argument("--network", default="Conv4_FC3",
                    help="Autoencoder network type. (default=Conv_4_FC_3). "
                         "Also, you can try training from scratch using VoxResNet and AllConvNet3D")
parser.add_argument('--selection', default="best_acc", choices=["best_acc", "best_loss"],
                    help="Evaluate the model performance based on which criterior")


# Computational issues
parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument('--gpu', default=False, action='store_true',
                    help='Uses gpu instead of cpu if cuda is available')


def main(options):

    # Initialize the model
    model = create_model(options.network, options.gpu)
    transformations = transforms.Compose([MinMaxNormalization()])

    print("Running for test the performances of the trained model,"
          "users should be responsible to take the right pretrained model")

    data_test = MRIDataset_patch_hippocampus(options.caps_directory, options.diagnosis_tsv_path,
                                             transformations=transformations)

    test_loader = DataLoader(data_test,
                             batch_size=options.batch_size,
                             shuffle=False,
                             num_workers=options.num_workers,
                             drop_last=True,
                             pin_memory=True)

    # Loop on all available folds
    fold_dirs = os.listdir(os.path.join(options.output_dir, 'best_model_dir'))
    for fold_dir in fold_dirs:
        fold = int(fold_dir[-1])

        # load the best trained model during the training
        model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', fold_dir,
                                                           'CNN', str(options.selection)),
                                       gpu=options.gpu, filename='model_best.pth.tar')
        model.eval()

        print("The best model was saved during training from fold %d at the %d -th epoch" % (fold, best_epoch))
        print("Please check if the model has been already severly overfitted at the best epoch by tensorboardX!")

        subjects, y_ground, y_hat, proba, accuracy_batch_mean = test(model, test_loader, options)
        print("Patch level balanced accuracy is %f" % (accuracy_batch_mean))

        # write the information of subjects and performances into tsv files.
        # Hard voting
        hard_voting_to_tsvs(options.output_dir, fold, subjects, y_ground, y_hat, proba, mode='test')

        # Soft voting
        soft_voting_to_tsvs(options.output_dir, fold, mode='test')


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
