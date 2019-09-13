import argparse
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import copy
import torch
import os
import numpy as np
from time import time

from .utils import MRIDataset_slice, train, test, slice_level_to_tsvs, soft_voting_to_tsvs
from tools.deep_learning import EarlyStopping, save_checkpoint, commandline_to_json, create_model, load_model
from tools.deep_learning.data import load_data, MinMaxNormalization

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for Pytorch 2D CNN, The input MRI's dimension is 169*208*179 after cropping")

# Mandatory arguments
parser.add_argument("caps_directory", type=str,
                    help="Path to the caps of image processing pipeline of DL")
parser.add_argument("diagnosis_tsv_path", type=str,
                    help="Path to tsv file of the population based on the diagnosis tsv files. "
                         "To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("output_dir", type=str,
                    help="Path to store the classification outputs and the tsv files containing the performances.")


parser.add_argument("--mri_plane", default=0, type=int,
                    help='Which coordinate axis to take for slicing the MRI. 0 is for sagittal, 1 is for coronal and '
                         '2 is for axial direction, respectively ')
parser.add_argument('--baseline', default=False, action="store_true",
                    help="Using baseline scans or all available longitudinal scans for training")

# Train argument
parser.add_argument("--network", default="resnet18",
                    help="Deep network type. Only ResNet was designed for training from scratch.")
parser.add_argument("--diagnoses", default=["AD", "CN"], type=str, nargs="+",
                    help="Labels for any binary task")

parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--split", default=None, type=int,
                    help="Define a specific fold in the k-fold, this is very useful to find the optimal model, where you do not want to run your k-fold validation")
parser.add_argument("--epochs", default=1, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument("--gpu", default=False, action="store_true",
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument("--num_workers", default=0, type=int,
                    help='the number of batch being loaded in parallel')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--selection_threshold', default=None, type=float,
                    help='Threshold on the balanced accuracies to compute the subject_level performance '
                         'only based on slices with balanced accuracy > threshold.')
parser.add_argument('--prepare_dl', default=False, action="store_true",
                    help="If True the outputs of preprocessing prepare_dl are used, else the whole MRI is loaded.")

# early stopping arguments
parser.add_argument("--patience", type=int, default=10,
                    help="tolerated epochs without improving for early stopping.")
parser.add_argument("--tolerance", type=float, default=0,
                    help="Tolerance of magnitude of performance after each epoch.")


def main(options):

    # Initialize the model
    print('Do transfer learning with existed model trained on ImageNet!\n')
    print('The chosen network is %s !' % options.network)

    model = create_model(options.network, options.gpu)
    trg_size = (224, 224)  # most of the imagenet pretrained model has this input size

    # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB
    # images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in
    # to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    transformations = transforms.Compose([MinMaxNormalization(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(trg_size),
                                          transforms.ToTensor()])

    # calculate the time consummation
    total_time = time()
    init_state = copy.deepcopy(model.state_dict())

    if options.split is None:
        fold_iterator = range(options.n_splits)
    else:
        fold_iterator = [options.split]

    for fi in fold_iterator:

        training_tsv, valid_tsv = load_data(options.diagnosis_tsv_path, options.diagnoses, fi,
                                            n_splits=options.n_splits, baseline=options.baseline)

        print("Running for the %d-th fold" % fi)

        data_train = MRIDataset_slice(options.caps_directory, training_tsv, transformations=transformations,
                                      mri_plane=options.mri_plane, prepare_dl=options.prepare_dl)
        data_valid = MRIDataset_slice(options.caps_directory, valid_tsv, transformations=transformations,
                                      mri_plane=options.mri_plane, prepare_dl=options.prepare_dl)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=True,
                                  num_workers=options.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  pin_memory=True)

        # chosen optimizer for back-propagation
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate, weight_decay=options.weight_decay)
        model.load_state_dict(init_state)

        # Binary cross-entropy loss
        loss = torch.nn.CrossEntropyLoss()

        # parameters used in training
        best_accuracy = 0.0
        best_loss_valid = np.inf

        writer_train_batch = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi,
                                                                 "train_batch")))
        writer_train_all_data = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi,
                                                                    "train_all_data")))
        writer_valid = SummaryWriter(log_dir=(os.path.join(options.output_dir, "log_dir", "fold_%i" % fi, "valid")))

        # initialize the early stopping instance
        early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)

        for epoch in range(options.epochs):
            print("At %i-th epoch." % epoch)

            # train the model
            train_df, acc_mean_train, loss_batch_mean_train, global_step \
                = train(model, train_loader, options.gpu, loss, optimizer, writer_train_batch, epoch,
                        model_mode='train', selection_threshold=options.selection_threshold)

            # calculate the accuracy with the whole training data for subject level balanced accuracy
            train_all_df, acc_mean_train_all, loss_batch_mean_train_all, _\
                = train(model, train_loader, options.gpu, loss, optimizer, writer_train_all_data, epoch,
                        model_mode='valid', selection_threshold=options.selection_threshold)
            print("For training, subject level balanced accuracy is %f at the end of epoch %d" % (acc_mean_train_all, epoch))

            # at then end of each epoch, we validate one time for the model with the validation data
            valid_df, acc_mean_valid, loss_batch_mean_valid, _ =\
                train(model, valid_loader, options.gpu, loss, optimizer, writer_valid, epoch,
                      model_mode='valid', selection_threshold=options.selection_threshold)
            print("For validation, subject level balanced accuracy is %f at the end of epoch %d" % (acc_mean_valid, epoch))

            # save the best model based on the best loss and accuracy
            acc_is_best = acc_mean_valid > best_accuracy
            best_accuracy = max(best_accuracy, acc_mean_valid)
            loss_is_best = loss_batch_mean_valid < best_loss_valid
            best_loss_valid = min(loss_batch_mean_valid, best_loss_valid)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'loss': loss_batch_mean_valid,
                'accuracy': acc_mean_valid,
                'optimizer': optimizer.state_dict(),
                'global_step': global_step},
                acc_is_best, loss_is_best,
                os.path.join(options.output_dir, "best_model_dir", "fold_" + str(fi), "CNN"))

            # try early stopping criterion
            if early_stopping.step(loss_batch_mean_valid) or epoch == options.epochs - 1:
                print("By applying early stopping or at the last epoch defined by user, "
                      "the training is stopped at %d-th epoch" % epoch)

                break

        # Final evaluation for all criteria
        for selection in ['best_loss', 'best_acc']:
            model, best_epoch = load_model(model, os.path.join(options.output_dir, 'best_model_dir', 'fold_%i' % fi,
                                                               'CNN', str(selection)),
                                           gpu=options.gpu, filename='model_best.pth.tar')

            train_df, metrics_train = test(model, train_loader, options.gpu, loss)
            valid_df, metrics_valid = test(model, valid_loader, options.gpu, loss)

            # write the information of subjects and performances into tsv files.
            slice_level_to_tsvs(options.output_dir, train_df, metrics_train, fi,
                                dataset='train', selection=selection)
            slice_level_to_tsvs(options.output_dir, valid_df, metrics_valid, fi,
                                dataset='validation', selection=selection)

            soft_voting_to_tsvs(options.output_dir, fi, dataset='train', selection=selection,
                                selection_threshold=options.selection_threshold)
            soft_voting_to_tsvs(options.output_dir, fi, dataset='validation', selection=selection,
                                selection_threshold=options.selection_threshold)

        del optimizer
        torch.cuda.empty_cache()

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)


if __name__ == "__main__":
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, "CNN")
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
