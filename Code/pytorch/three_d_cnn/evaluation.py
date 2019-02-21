import argparse
import torch
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import *
from resume import parse_model_name

parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

# Mandatory arguments
parser.add_argument("diagnosis_path", type=str,
                    help="Path to the folder containing the tsv files of the population.")
parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")

# Data Management
parser.add_argument("--data_path", default="linear", choices=["linear", "dartel", "mni"], type=str,
                    help="Defines the path to data in CAPS.")
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
parser.add_argument("--graphics", action="store_true", default=False,
                    help="Plot graphics.")

# Computational ressources
parser.add_argument("--num_workers", '-w', default=8, type=int,
                    help='the number of batch being loaded in parallel')


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    options = parse_model_name(options.model_path, options)
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

    # Loop on all folds trained
    CNN_dir = os.path.join(options.model_path, 'best_model_dir', 'CNN')
    folds_dir = os.listdir(CNN_dir)
    for fold_dir in folds_dir:
        split = int(fold_dir[-1])
        print("Fold " + str(split))
        model = eval(options.model)()
        if options.gpu:
            model = model.cuda()
        options.batch_size = 2  # To test on smaller GPU

        criterion = torch.nn.CrossEntropyLoss()

        if options.selection == 'loss':
            model_dir = os.path.join(CNN_dir, fold_dir, 'best_loss')
            folder_name = 'best_loss'
        else:
            model_dir = os.path.join(CNN_dir, fold_dir, 'best_acc')
            folder_name = 'best_acc'

        best_model, best_epoch = load_model(model, model_dir,
                                            filename='model_best.pth.tar')

        training_tsv, valid_tsv = load_data(options.diagnosis_path, options.diagnoses,
                                            split, options.n_splits, options.baseline, options.data_path)

        if options.minmaxnormalization:
            transformations = MinMaxNormalization()
        else:
            transformations = None

        data_train = MRIDataset(options.input_dir, training_tsv, options.data_path, transform=transformations)
        data_valid = MRIDataset(options.input_dir, valid_tsv, options.data_path, transform=transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        metrics_train, loss_train, train_df = test(best_model, train_loader, options.gpu, criterion,
                                                   verbose=False, full_return=True)
        metrics_valid, loss_valid, valid_df = test(best_model, valid_loader, options.gpu, criterion,
                                                   verbose=False, full_return=True)

        acc_train, sen_train, spe_train = metrics_train['balanced_accuracy'] * 100, metrics_train['sensitivity'] * 100,\
                                          metrics_train['specificity'] * 100
        acc_valid, sen_valid, spe_valid = metrics_valid['balanced_accuracy'] * 100, metrics_valid['sensitivity'] * 100, \
                                          metrics_valid['specificity'] * 100
        print("Training, acc %f, loss %f, sensibility %f, specificity %f"
              % (acc_train, loss_train, sen_train, spe_train))
        print("Validation, acc %f, loss %f, sensibility %f, specificity %f"
              % (acc_valid, loss_valid, sen_valid, spe_valid))

        evaluation_path = path.join(options.model_path, 'performances', fold_dir)
        if not path.exists(path.join(evaluation_path, folder_name)):
            os.makedirs(path.join(evaluation_path, folder_name))

        text_file = open(path.join(evaluation_path, 'evaluation_' + options.selection + '.txt'), 'w')
        text_file.write('Best epoch: %i \n' % best_epoch)
        text_file.write('Accuracy on training set: %.2f %% \n' % acc_train)
        text_file.write('Loss on training set: %f \n' % loss_train)
        text_file.write('Sensitivities on training set: %.2f %%, %.2f %% \n' % (sen_train, sen_train))
        text_file.write('Specificities on training set: %.2f %%, %.2f %% \n' % (spe_train, spe_train))

        text_file.write('Accuracy on validation set: %.2f %% \n' % acc_valid)
        text_file.write('Loss on validation set: %f \n' % loss_valid)
        text_file.write('Sensitivities on validation set: %.2f %%, %.2f %% \n' % (sen_valid, sen_valid))
        text_file.write('Specificities on validation set: %.2f %%, %.2f %% \n' % (spe_valid, spe_valid))

        text_file.close()

        train_df.to_csv(path.join(evaluation_path, folder_name, 'train_subject_level_result.tsv'), sep='\t', index=False)
        valid_df.to_csv(path.join(evaluation_path, folder_name, 'valid_subject_level_result.tsv'), sep='\t', index=False)

        # Save all metrics except confusion matrix
        del metrics_train['confusion_matrix']
        pd.DataFrame(metrics_train, index=[0]).to_csv(path.join(evaluation_path, folder_name,
                                                                'train_subject_level_metrics.tsv'),
                                                      sep='\t', index=False)
        del metrics_valid['confusion_matrix']
        pd.DataFrame(metrics_valid, index=[0]).to_csv(path.join(evaluation_path, folder_name,
                                                                'valid_subject_level_metrics.tsv'),
                                                      sep='\t', index=False)

        # Graphic part
        if options.graphics:
            import matplotlib.pyplot as plt

            plt.switch_backend('agg')

            training_df = pd.read_csv(path.join(options.model_path, 'log_dir', 'CNN', fold_dir, 'training.tsv'), sep='\t')
            epochs = training_df.epoch.values
            iterations = training_df.iteration.values
            iterations = iterations / np.max(iterations)
            x = epochs + iterations

            plt.title('Fold ' + str(split))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(x, training_df.mean_loss_train.values, color='orange', label='training')
            plt.plot(x, training_df.mean_loss_valid.values, color='blue', label='validation')
            plt.legend()
            plt.savefig(path.join(evaluation_path, 'loss.png'))
            plt.close()

            plt.title('Fold ' + str(split))
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.plot(x, training_df.acc_train.values, color='orange', label='training')
            plt.plot(x, training_df.acc_valid.values, color='blue', label='validation')
            plt.legend()
            plt.savefig(path.join(evaluation_path, 'accuracy.png'))
            plt.close()
