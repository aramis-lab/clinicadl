# coding: utf8

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import os

from clinicadl.slice_level.utils import MRIDataset_slice, test, slice_level_to_tsvs, soft_voting_to_tsvs
from clinicadl.tools.deep_learning import create_model, load_model
from clinicadl.tools.deep_learning.data import MinMaxNormalization, load_data_test, load_data

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D slice-level CNN for test the trained classifiers")

# Mandatory arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to the tsv containing all the test dataset")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs, and the tsv files containing the performances.")

# Data arguments
parser.add_argument("--diagnoses", default=["sMCI", "pMCI"], type=str, nargs="+",
                    help="Labels based on binary classification.")
parser.add_argument('--dataset', default="validation", type=str,
                    help="If the evaluation on the validation set is wanted, must be set to 'validation'. "
                         "Otherwise must be named with the form 'test-cohort_name'.")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Default behaviour will run all splits, else only the splits specified will be run.")
parser.add_argument("--mri_plane", default=0, type=int,
                    help='Which coordinate axis to take for slicing the MRI. 0 is for sagittal, '
                         '1 is for coronal and 2 is for axial direction, respectively ')
parser.add_argument('--selection', default="best_acc", choices=["best_acc", "best_loss"],
                    help="Evaluate the model performance based on which criterion.")
parser.add_argument('--prepare_dl', default=False, action="store_true",
                    help="If True the outputs of preprocessing prepare_dl are used, else the whole MRI is loaded.")

# test argument
parser.add_argument("--network", default="resnet18",
                    help="Deep network type. Only ResNet was designed for training from scratch.")
parser.add_argument('--selection_threshold', default=None, type=float,
                    help='Threshold on the balanced accuracies to compute the subject_level performance '
                         'only based on patches with balanced accuracy > threshold.')

# Computational issues
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--gpu", default=False, action="store_true",
                    help="If use gpu or cpu. Empty implies cpu usage.")
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    # Initialize the model
    print('Do transfer learning with existed model trained on ImageNet.')

    model = create_model(options.network, options.gpu)
    trg_size = (224, 224)  # most of the imagenet pretrained model has this input size

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
    # images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in
    # to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])
    # Define loss and optimizer
    loss = torch.nn.CrossEntropyLoss()

    if options.split is None:
        fold_iterator = range(options.n_splits)
    else:
        fold_iterator = [options.split]

    # Loop on folds
    for fi in fold_iterator:
        print("Fold %i" % fi)

        if options.dataset == 'validation':
            _, test_df = load_data(options.diagnosis_tsv_path, options.diagnoses, fi,
                                   n_splits=options.n_splits, baseline=True)
        else:
            test_df = load_data_test(options.diagnosis_tsv_path, options.diagnoses)

        data_test = MRIDataset_slice(options.caps_directory, test_df,
                                     transformations=transformations, mri_plane=options.mri_plane,
                                     prepare_dl=options.prepare_dl)

        test_loader = DataLoader(data_test,
                                 batch_size=options.batch_size,
                                 shuffle=False,
                                 num_workers=options.num_workers,
                                 pin_memory=True)

        # load the best trained model during the training
        model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', "fold_%i" % fi,
                                                           'CNN', str(options.selection)),
                                       gpu=options.gpu, filename='model_best.pth.tar')

        results_df, metrics = test(model, test_loader, options.gpu, loss)
        print("Slice level balanced accuracy is %f" % (metrics['balanced_accuracy']))

        slice_level_to_tsvs(options.output_dir, results_df, metrics, fi, options.selection, dataset=options.dataset)

        # Soft voting
        soft_voting_to_tsvs(options.output_dir, fi, selection=options.selection, dataset=options.dataset,
                            selection_threshold=options.selection_threshold)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
