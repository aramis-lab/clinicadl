import argparse
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
from os import path

package_path = path.abspath(path.join(path.abspath(path.join(path.abspath(path.join(path.abspath(path.join(sys.argv[0], os.pardir)), os.pardir)), os.pardir)), os.pardir))
sys.path.append(package_path)

from classifiers.three_d_cnn.patch_level.classification_utils import *

from tools.deep_learning.data import MinMaxNormalization
from tools.deep_learning.models import create_model


__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for test of hippocampus approach")

## Data arguments
parser.add_argument("--caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv_path", default='/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/ROI_based/Finished_exp/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch200_baseline_hippocampus50_with_es_MedIA',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")

parser.add_argument('--best_model_fold', default=4,
                    help="Use the best from the which fold of training")
parser.add_argument('--gpu', default=False, action='store_true',
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--network", default="Conv_4_FC_3", choices=["Conv_4_FC_3", "Conv_7_FC_2", "Conv_3_FC_2"],
                    help="Autoencoder network type. (default=Conv_4_FC_3). Also, you can try training from scratch using VoxResNet and AllConvNet3D")
parser.add_argument('--best_model_criteria', default="best_acc", choices=["best_acc", "best_loss"],
                    help="Evaluate the model performance based on which criterior")


def main(options):

    # Initial the model
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

    # load the best trained model during the training
    model, best_global_step, best_epoch, best_predict = load_model_test(model, os.path.join(options.output_dir, 'best_model_dir',
                                                                                   "fold_" + str(options.best_model_fold), 'CNN', str(options.best_model_criteria)),
                                                                      filename='model_best.pth.tar')

    print("The best model was saved during training from fold %d at the %d -th epoch at the %d -th global step" % (int(options.best_model_fold), int(best_epoch), int(best_global_step)))
    print("Please check if the model has been already severly overfitted at the best epoch by tensorboardX!")

    subjects, y_ground, y_hat, proba, accuracy_batch_mean = test(model, test_loader, options)
    print("Patch level balanced accuracy is %f" % (accuracy_batch_mean))

    # write the information of subjects and performances into tsv files.
    # Hard voting
    hard_voting_to_tsvs(options.output_dir, options.best_model_fold, subjects, y_ground, y_hat, proba, mode='test')

    # Soft voting
    soft_voting_to_tsvs(options.output_dir, options.best_model_fold, mode='test')


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
