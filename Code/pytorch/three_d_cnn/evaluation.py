import argparse
import torch
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import *
from resume import parse_model_name

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D AE pretraining")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")

# Data Management
parser.add_argument("--selection", default="loss", type=str, choices=['loss', 'accuracy'],
                    help="Loads the model selected on minimal loss or maximum accuracy on validation.")
parser.add_argument("--diagnoses", "-d", default=["AD", "CN"], nargs='+', type=str,
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--baseline", default=False, action="store_true",
                    help="Use only baseline data instead of all scans available")
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--gpu", action="store_true", default=False,
                    help="if True computes the visualization on GPU")
parser.add_argument("--minmaxnormalization", "-n", default=False, action="store_true",
                    help="Performs MinMaxNormalization for visualization")
parser.add_argument("--feature_maps", "-fm", default=False, action="store_true",
                    help="Performs feature maps extraction and visualization")
parser.add_argument("--filters", "-f", default=False, action="store_true",
                    help="Performs the visualization of filters by optimizing images which maximally activate"
                         "each filter.")


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    options, run = parse_model_name(options.model_path, options)
    # Check if model is implemented
    import model
    import inspect

    choices = []
    for name, obj in inspect.getmembers(model):
        if inspect.isclass(obj):
            choices.append(name)

    if options.model not in choices:
        raise NotImplementedError(
            'The model wanted %s has not been implemented in the module model.py' % options.model)

    model = eval(options.model)()
    if options.gpu:
        model = model.cuda()
    options.batch_size = 2  # To test on smaller GPU

    criterion = torch.nn.CrossEntropyLoss()
    best_model, best_epoch = load_model(model, options.model_path,
                                        filename='model_best_' + options.selection + '.pth.tar')

    training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                        options.split, options.n_splits, options.baseline)

    if options.minmaxnormalization:
        transformations = MinMaxNormalization()
    else:
        transformations = None

    data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations)
    data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations)

    # Use argument load to distinguish training and testing
    train_loader = DataLoader(data_train,
                              batch_size=options.batch_size,
                              shuffle=options.shuffle,
                              num_workers=options.num_workers,
                              drop_last=False
                              )

    valid_loader = DataLoader(data_valid,
                              batch_size=options.batch_size,
                              shuffle=False,
                              num_workers=options.num_workers,
                              drop_last=False
                              )

    acc_train, loss_train, sen_train, spe_train = test(best_model, train_loader, options.gpu, criterion,
                                                       verbose=False, full_return=True)
    acc_valid, loss_valid, sen_valid, spe_valid = test(best_model, valid_loader, options.gpu, criterion,
                                                       verbose=False, full_return=True)
    print("Training, acc %f, loss %f, sensibility %f, specificity %f"
          % (acc_train, loss_train, sen_train[0], spe_train[0]))
    print("Validation, acc %f, loss %f, sensibility %f, specificity %f"
          % (acc_valid, loss_valid, sen_valid[0], spe_valid[0]))

    text_file = open(path.join(options.model_path, 'evaluation_' + options.selection + '.txt'), 'w')
    text_file.write('Best epoch: %i \n' % best_epoch)
    text_file.write('Accuracy on training set: %.2f %% \n' % acc_train)
    text_file.write('Loss on training set: %f \n' % loss_train)
    text_file.write('Sensitivities on training set: %.2f %%, %.2f %% \n' % (sen_train[0], sen_train[1]))
    text_file.write('Specificities on training set: %.2f %%, %.2f %% \n' % (spe_train[0], spe_train[1]))

    text_file.write('Accuracy on training set: %.2f %% \n' % acc_valid)
    text_file.write('Loss on training set: %f \n' % loss_valid)
    text_file.write('Sensitivities on training set: %.2f %%, %.2f %% \n' % (sen_valid[0], sen_valid[1]))
    text_file.write('Specificities on training set: %.2f %%, %.2f %% \n' % (spe_valid[0], spe_valid[1]))

    text_file.close()
