import argparse
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from classification_utils import *
from model import *
import os

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D patch-level multi-CNN for test the trained classifiers")

## data argument
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv", default='/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv',
                           help="Path to the tsv containing all the test dataset")
parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/Hao_results/sMCI_pMCI/patch_level/pytorch_AE_Conv_4_FC_2_bs4_lr_e5_only_finetuning_epoch100_longitudinal_hippocampus50_es10_baselineCNN_MedIA',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument('--best_model_criteria', default="best_acc", choices=["best_acc", "best_loss"],
                    help="Evaluate the model performance based on which criterior")
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument('--mode', default="test", choices=["test", "valid"],
                    help="Evaluate or test")

## train argument
# transfer learning
parser.add_argument("--network", default="Conv_4_FC_3", choices=["Conv_4_FC_3", "Conv_7_FC_2", "Conv_3_FC_2"],
                    help="Autoencoder network type. (default=Conv_4_FC_3). Also, you can try training from scratch using VoxResNet and AllConvNet3D")
parser.add_argument("--num_cnn", default=36, type=int,
                    help="How many CNNs we want to train in a patch-wise way. By default, we train each patch from all subjects for one CNN")
parser.add_argument("--diagnoses_list", default=["sMCI", "pMCI"], type=str,
                    help="Labels based on binary classification")
parser.add_argument('--n_fold', default=0,
                    help="which fold to be test")
parser.add_argument("--use_gpu", default=True, type=bool,
                    help="If use gpu or cpu. Empty implies cpu usage.")
parser.add_argument("--diagnosis_tsv_path", default='/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")

## TODO; check the behavior of default for bool in argparser

def main(options):
    # Initial the model
    model = eval(options.network)()
    ## need to normalized the value to [0, 1]
    transformations = transforms.Compose([NormalizeMinMax()])

    ## get the test accuracy for all the N classifiers
    for n in range(options.num_cnn):
        if options.mode == 'test':
            dataset = MRIDataset_patch_by_index(options.caps_directory, options.diagnosis_tsv, options.patch_size,
                                                   options.patch_stride, n, transformations=transformations)

            writer = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(options.n_fold), "cnn-" + str(n), "test")))

        else:
            _, _, _, valid_tsv = load_split_by_diagnosis(options, options.n_fold, baseline_or_longitudinal='baseline', autoencoder=False)

            dataset = MRIDataset_patch_by_index(options.caps_directory, valid_tsv, options.patch_size,
                                                   options.patch_stride, n, transformations=transformations)

            writer = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(options.n_fold), "cnn-" + str(n), "valid-best")))

        data_loader = DataLoader(dataset,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=options.num_workers,
                                 drop_last=True,
                                 pin_memory=True)

        # load the best trained model during the training
        model_updated, best_global_step, best_epoch, best_predict = load_model_test(model, os.path.join(options.output_dir, 'best_model_dir',
                                                                                       "fold_" + str(options.n_fold), 'cnn-' + str(n), options.best_model_criteria),
                                                                          filename='model_best.pth.tar')

        print("The best model was saved during training from fold %d at the %d -th epoch at the %d -th global step" % (int(options.n_fold), int(best_epoch), int(best_global_step)))

        ## Decide to use gpu or cpu to train the model
        if options.use_gpu == False:
            model_updated.cpu()
        else:
            print("Using GPU")
            model_updated.cuda()

        # Define loss and optimizer
        loss = torch.nn.CrossEntropyLoss()
        # chosen optimer for back-propogation
        optimizer = eval("torch.optim.Adam")(filter(lambda x: x.requires_grad, model_updated.parameters()), 0.001,
                                                             weight_decay=0.001)

        # test_subjects, y_ground_test, y_hat_test, test_proba, acc_mean_test, global_step, loss_batch_mean_test = train(
        #     model_updated, data_loader, options.use_gpu, loss, optimizer, writer, 0, options.n_fold, model_mode='valid',
        #     global_step=0)
        # print("Subject level balanced accuracy is %f" % (acc_mean_test))

        subjects, y_ground, y_hat, proba, accuracy_batch_mean = test(model_updated, data_loader, options)
        print("Patch level balanced accuracy is %f" % (accuracy_batch_mean))

        # write the test results into the tsv files
        hard_voting_to_tsvs(options.output_dir, options.n_fold, subjects, y_ground, y_hat, proba, mode=options.mode, patch_index=n)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)

