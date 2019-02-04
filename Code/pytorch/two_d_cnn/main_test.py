import argparse
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from classification_utils import *
from model import *
import copy
from time import time

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D slice-level CNN for test the trained classifiers")

## data argument
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv',
                           help="Path to the tsv containing all the test dataset")
parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/2d_slice/ResNet_tl/longitudinal/5_fold/pytorch_resnet18_tl_fintune_lastResBlock_top_last1fc_dropout0.8_lr10-6_bs32_ep50_wd10-4_baseline',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--data_type", default="from_slice", choices=["from_MRI", "from_slice"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
parser.add_argument("--mri_plane", default=0, type=int,
                    help='Which coordinate axis to take for slicing the MRI. 0 is for saggital, 1 is for coronal and 2 is for axial direction, respectively ')
parser.add_argument('--image_processing', default="LinearReg", choices=["LinearReg", "Segmented"],
                    help="The output of which image processing pipeline to fit into the network. By defaut, using the raw one with only linear registration, otherwise, using the output of spm pipeline of Clinica")
parser.add_argument('--best_model_fold', default=0,
                    help="Use the best from the which fold of training")
parser.add_argument('--best_model_criteria', default="best_acc", choices=["best_acc", "best_loss"],
                    help="Evaluate the model performance based on which criterior")

## train argument
parser.add_argument("--transfer_learning", default=True, type=bool, help="If the trained model was trained by transfer learning")
parser.add_argument("--network", default="ResNet", choices=["AlexNet", "ResNet", "LeNet", "AllConvNet", "Vgg16", "DenseNet161", "InceptionV3", "AlexNetonechannel"],
                    help="Deep network type. Only ResNet was designed for training from scratch.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--use_gpu", default=True, type=bool,
                    help="If use gpu or cpu. Empty implies cpu usage.")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')

## TODO; check the behavior of default for bool in argparser

def main(options):

    # Initial the model
    if options.transfer_learning == True:
        print('Do transfer learning with existed model trained on ImageNet!\n')
        print('The chosen network is %s !' % options.network)

        try:
            model = eval(options.network)()
            if options.network == "InceptionV3":
                trg_size = (299, 299)
            else:
                trg_size = (224, 224) # most of the imagenet pretrained model has this input size
        except:
            raise Exception('The model has not been implemented or has bugs in to model implementation')

        ## All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
        # images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in
        # to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        transformations = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(trg_size),
                                              transforms.ToTensor()])
    else:
        print('Train the model from scratch!')
        print('The chosen network is %s !' % options.network)

        try:
            model = eval(options.network)(mri_plane=options.mri_plane)
        except:
            raise Exception('The model has not been implemented')
        transformations = None

    print("Running for test the performances of the trained model, users should be responsible to take the right right pretrained model")

    data_test = MRIDataset_slice(options.caps_directory, options.diagnosis_tsv, transformations=transformations, transfer_learning=options.transfer_learning, mri_plane=options.mri_plane, data_type=options.data_type, image_processing=options.image_processing)

    test_loader = DataLoader(data_test,
                             batch_size=options.batch_size,
                             shuffle=False,
                             num_workers=options.num_workers,
                             drop_last=True,
                             pin_memory=True)


    # load the best trained model during the training
    model, best_global_step, best_epoch = load_model_test(model, os.path.join(options.output_dir, 'best_model_dir',
                                                                                   "fold_" + str(options.best_model_fold), str(options.best_model_criteria)),
                                                                      filename='model_best.pth.tar')

    print("The best model was saved during training from fold %d at the %d -th epoch at the %d -th global step" % (options.best_model_fold, best_epoch, best_global_step))
    print("Please check if the model has been already severly overfitted at the best epoch by tensorboardX!")

    ## Decide to use gpu or cpu to train the model
    if options.use_gpu == False:
        use_cuda = False
        model.cpu()
    else:
        print("Using GPU")
        use_cuda = True
        model.cuda()

    subjects, y_ground, y_hat, accuracy_batch_mean = test(model, test_loader, use_cuda)
    print("Slice level balanced accuracy is %f" % (accuracy_batch_mean))

    ### write the information of subjects and performances into tsv files.
    ## TODO soft voting system
    fold_subjects_df_train, results_train = results_to_tsvs(options.output_dir, options.best_model_fold, subjects, y_ground, y_hat, mode='test')

if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
