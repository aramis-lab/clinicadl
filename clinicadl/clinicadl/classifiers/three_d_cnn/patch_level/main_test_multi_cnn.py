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
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv',
                           help="Path to the tsv containing all the test dataset")
parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/3d_patch/AD_CN/longitudinal/final_results/pytorch_AE_Conv_4_FC_3_bs32_lr_e5_only_finetuning_epoch20_ps_50_ss_50_baseline_all_patch_backup_multiCNN',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--data_type", default="from_patch", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
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

## train argument
# transfer learning
parser.add_argument("--network", default="Conv_4_FC_3", choices=["Conv_4_FC_3", "Conv_7_FC_2", "Conv_3_FC_2"],
                    help="Autoencoder network type. (default=Conv_4_FC_3). Also, you can try training from scratch using VoxResNet and AllConvNet3D")
parser.add_argument("--num_cnn", default=36, type=int,
                    help="How many CNNs we want to train in a patch-wise way. By default, we train each patch from all subjects for one CNN")
parser.add_argument("--diagnoses_list", default=["AD", "CN"], type=str,
                    help="Labels based on binary classification")
parser.add_argument('--n_fold', default=0,
                    help="which fold to be test")

parser.add_argument("--use_gpu", default=True, type=bool,
                    help="If use gpu or cpu. Empty implies cpu usage.")

## TODO; check the behavior of default for bool in argparser

def main(options):
    # Initial the model
    model = eval(options.network)()
    ## need to normalized the value to [0, 1]
    transformations = transforms.Compose([NormalizeMinMax()])

    ## get the weight for soft voting system from all validation acc from all N classifiers

    weight_list = weight_by_validation_acc(model, options)

    ## get the test accuracy for all the N classifiers
    for n in range(options.num_cnn):
        data_test = MRIDataset_patch_by_index(options.caps_directory, options.diagnosis_tsv, options.patch_size,
                                               options.patch_stride, n, transformations=transformations,
                                               data_type=options.data_type)
        test_loader = DataLoader(data_test,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=options.num_workers,
                                 drop_last=True,
                                 pin_memory=True)
        writer_test = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(options.n_fold), "cnn-" + str(n), "test")))


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

        test_subjects, y_ground_test, y_hat_test, test_proba, acc_mean_test, global_step, loss_batch_mean_test = train(
            model_updated, test_loader, options.use_gpu, loss, optimizer, writer_test, 0, options.n_fold, model_mode='valid',
            global_step=0)
        print("Subject level balanced accuracy is %f" % (acc_mean_test))

        ## write the test results into the tsv files
        hard_voting_to_tsvs(options.output_dir, options.n_fold, test_subjects, y_ground_test, y_hat_test, test_proba, mode='test', patch_index=n)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)

