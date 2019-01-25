import argparse
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import *

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D AE pretraining")

# Mandatory arguments
parser.add_argument("-dt", "--diagnosis_path", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv',
                           help="Path to tsv file of the population. To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("-od", "--result_path", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/pytorch_ae_elina',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("-id", "--input_dir", default='/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL',
                           help="Path to the caps of image processing pipeline of DL")
parser.add_argument("-md", "--model", type=str, default='Conv_3', choices=["Conv_3", "Conv_4", "Test", "Test_nobatch", "Rieke", "Test2", 'Optim'],
                    help="model selected")

# Data Management
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument('--accumulation_steps', '-asteps', default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--runs", default=1, type=int,
                    help="Number of runs with the same training / validation split.")
parser.add_argument("--diagnoses", default=["AD", "CN"], nargs='+', type=str,
                    help="Take all the subjects possible for autoencoder training")
parser.add_argument("--baseline", action="store_true", default=False,
                    help="if True only the baseline is used")
parser.add_argument("--visualization", action='store_true', default=False,
                    help='Chooses if visualization is done on AE pretraining')

# Training arguments
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--greedy_learning", action="store_true", default=False,
                    help="Optimize with greedy layer-wise learning")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')
parser.add_argument('--num_threads', type=int, default=1,
                    help='Number of threads used.')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')


def main(options):

    options.transfer_learning_rate = options.learning_rate
    options.transfer_learning_epochs = options.epochs

    check_and_clean(options.result_path)
    torch.set_num_threads(options.num_threads)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    transformations = None

    total_time = time()
    # Training the autoencoder based on the model
    model = eval(options.model)()
    criterion = torch.nn.MSELoss()

    # training_tsv, valid_tsv = load_autoencoder_data(options.diagnosis_path, options.diagnoses)
    training_tsv, valid_tsv = load_split(options.diagnosis_path)

    data_train = MRIDataset(options.input_dir, training_tsv, transformations)
    data_valid = MRIDataset(options.input_dir, valid_tsv, transformations)

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

    for run in range(options.runs):
        print('Beginning run %i' % run)
        run_path = path.join(options.result_path, 'run' + str(run))

        if options.greedy_learning:
            greedy_learning(model, train_loader, valid_loader, criterion, options.gpu, run_path, options)

        else:
            decoder = AutoEncoder(model)
            ae_finetuning(decoder, train_loader, valid_loader, criterion, options.gpu, run_path, options)

            best_decoder = load_model(decoder, run_path)
            visualize_ae(best_decoder, train_loader, path.join(run_path, "train"), options.gpu)
            visualize_ae(best_decoder, valid_loader, path.join(run_path, "valid"), options.gpu)

    total_time = time() - total_time
    print('Total time', total_time)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
