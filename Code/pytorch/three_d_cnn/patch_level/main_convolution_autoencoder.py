import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from classification_utils import *
from model import *
import torchvision.transforms as transforms
import copy

__author__ = "Junhao Wen, Elina Thibeausutre"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for 3D convolutional autoencoder, the AE will be reconstructed based on the CNN that you choose")

## Data arguments
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("--diagnosis_tsv_path", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_diagnosis/test',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("--output_dir", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_ae_conv',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--data_type", default="from_patch", choices=["from_MRI", "from_patch"],
                    help="Use which data to train the model, as extract slices from MRI is time-consuming, we recommand to run the postprocessing pipeline and train from slice data")
parser.add_argument("--patch_size", default=50, type=int,
                    help="The patch size extracted from the MRI")
parser.add_argument("--patch_stride", default=50, type=int,
                    help="The stride for the patch extract window from the MRI")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Define a specific fold in the k-fold, this is very useful to find the optimal model, where you do not want to run your k-fold validation")
parser.add_argument('--baseline_or_longitudinal', default="baseline", choices=["baseline", "longitudinal"],
                    help="Using baseline scans or all available longitudinal scans for training")
parser.add_argument('--hippocampus_roi', default=False, type=bool,
                    help="If train the model using only hippocampus ROI")

# Training arguments
parser.add_argument("--network", default="Conv_3_FC_2", choices=["Conv_4_FC_2", "Conv_7_FC_2", "Conv_3_FC_2"],
                    help="Autoencoder network type. (default=Conv_4_FC_2)")
parser.add_argument("--ae_training_method", default="stacked_ae", choices=["layer_wise_ae", "stacked_ae"],
                    help="How to train the autoencoder, layer wise or train all AEs together")
parser.add_argument("--diagnoses_list", default=["AD", "CN", "MCI"], type=str,
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--epochs_layer_wise", default=1, type=int,
                    help="Epochs for layer-wise AE training")
parser.add_argument("--epochs_fine_tuning", default=1, type=int,
                    help="Epochs for fine tuning all the stacked AEs after greedy layer-wise training, or directly train the AEs together")
parser.add_argument("--learning_rate", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--use_gpu", default=True, type=bool,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--weight_decay", default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument("--accumulation_steps",  default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')

## visualization
parser.add_argument("--visualization", default=True, type=bool,
                    help='Chooses if visualization is done on AE pretraining')

def main(options):

    print('Start the training for stacked convolutional autoencoders, the optimal model be saved for future use!')
    try:
        model = eval(options.network)()
    except:
        raise Exception('The model has not been implemented or has bugs in the model codes')

    ## need to normalized the value to [0, 1]
    transformations = transforms.Compose([NormalizeMinMax()])

    ## the inital model weight and bias
    init_state = copy.deepcopy(model.state_dict())

    if options.split != None:
        print("Only run for a specific fold, meaning that you are trying to find your optimal model by exploring your training and validation data")
        options.n_splits = 1

    for fi in range(options.n_splits):

        # to set the split = 0
        if options.split != None:
            ## train seperately a specific fold during the k-fold, also good for the limitation of your comuptational power
            _, _, training_tsv, valid_tsv = load_split_by_diagnosis(options, options.split, baseline_or_longitudinal=options.baseline_or_longitudinal)
            fi = options.split
        else:
             _, _, training_tsv, valid_tsv = load_split_by_diagnosis(options, fi, baseline_or_longitudinal=options.baseline_or_longitudinal)

        print("Running for the %d -th fold" % fi)

        if options.hippocampus_roi:
            print("Only using hippocampus ROI")

            data_train = MRIDataset_patch_hippocampus(options.caps_directory, training_tsv, transformations=transformations)
            data_valid = MRIDataset_patch_hippocampus(options.caps_directory, valid_tsv, transformations=transformations)

        else:
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

        if fi != 0:
            model = eval(options.network)()
        model.load_state_dict(init_state)

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

        criterion = torch.nn.MSELoss()
        writer_train = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "layer_wise", "train")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "layer_wise", "valid")))
        writer_train_ft = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "fine_tine", "train")))
        writer_valid_ft = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_" + str(fi), "ConvAutoencoder", "fine_tine", "valid")))

        if options.ae_training_method == 'layer_wise_ae':
            ## TODO: check the memory accumulation during the 5-fold CV
            model, best_autodecoder = greedy_layer_wise_learning(model, train_loader, valid_loader, criterion, use_cuda, writer_train, writer_valid, writer_train_ft, writer_valid_ft, options, fi)
        else:
            model, best_autodecoder = stacked_ae_learning(model, train_loader, valid_loader, criterion, use_cuda,
                                                          writer_train_ft, writer_valid_ft,
                                                                 options, fi)

        ## save the graph and image
        # TODO bug to save the model graph for 3D patch, here is the discuss: https://github.com/lanpa/tensorboardX/issues/346
        # writer_train.add_graph(best_autodecoder, example_batch)

        if options.visualization:
            visualize_ae(best_autodecoder, example_batch, os.path.join(options.output_dir, "visualize", "fold_" + str(fi)))

        del best_autodecoder, train_loader, valid_loader, example_batch, criterion, model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    ## save the commind line arguments into a tsv file for tracing all different kinds of experiments
    commandline_to_jason(commandline, pretrain_ae=True)
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
