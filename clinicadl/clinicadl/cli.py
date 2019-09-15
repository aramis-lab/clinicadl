import argparse

from .tools.deep_learning.iotools import Parameters
from .tools.deep_learning import commandline_to_json
from .preprocessing.T1_preprocessing import preprocessing_t1w
from .preprocessing.T1_postprocessing import postprocessing_t1w
from .subject_level.train_autoencoder import train_autoencoder
from .subject_level.train_CNN import train_cnn
from .slice_level.train_CNN import train_slice

def preprocessing_t1w_func(args):
    wf = preprocessing_t1w(args.bids_directory, 
            args.caps_dir,
            args.tsv_file,
            args.ref_template,
            args.working_directory)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})

def extract_data_func(args):
    wf = postprocessing_t1w(args.caps_dir, 
            args.tsv_file,
            args.patch_size,
            args.stride_size,
            args.working_directory,
            args.extract_method,
            args.slice_direction,
            args.slice_mode)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})


# Function to dispatch training to corresponding function
def train_func(args):
    if args.mode=='subject' :
       if args.train_autoencoder :
           train_params_autoencoder = Parameters(args.tsv_path, 
                   args.output_dir, 
                   args.caps_dir, 
                   args.network)
           train_params_autoencoder.write(args.pretrained_path,
                   args.pretrained_difference,
                   args.preprocessing,
                   args.diagnoses,
                   args.baseline,
                   args.minmaxnormalization,
                   'random',
                   args.n_splits,
                   args.split,
                   args.accumulation_steps,
                   args.epochs,
                   args.learning_rate,
                   args.patience,
                   args.tolerance,
                   args.add_sigmoid,
                   'Adam',
                   0.0,
                   args.use_gpu,
                   args.batch_size,
                   args.evaluation_steps,
                   args.nproc)
           train_autoencoder(train_params_autoencoder)
       else:
           train_params_cnn = Parameters(args.tsv_path, 
                   args.output_dir, 
                   args.caps_dir, 
                   args.network)
           train_params_cnn.write(args.pretrained_path,
                   args.pretrained_difference,
                   args.preprocessing,
                   args.diagnoses,
                   args.baseline,
                   args.minmaxnormalization,
                   args.sampler,
                   args.n_splits,
                   args.split,
                   args.accumulation_steps,
                   args.epochs,
                   args.learning_rate,
                   args.patience,
                   args.tolerance,
                   args.add_sigmoid,
                   'Adam',
                   0.1,
                   args.use_gpu,
                   args.batch_size,
                   args.evaluation_steps,
                   args.nproc,
                   args.transfer_learning_path,
                   args.transfer_learning_autoencoder,
                   args.selection)
           train_cnn(train_params_cnn)
    elif args.mode=='slice':
        train_params_slice = Parameters(args.tsv_path, 
                args.output_dir, 
                args.caps_dir, 
                args.network)
        train_params_slice.write(args.pretrained_path,
                args.pretrained_difference,
                args.preprocessing,
                args.diagnoses,
                args.baseline,
                args.minmaxnormalization,
                args.sampler,
                args.n_splits,
                args.split,
                args.accumulation_steps,
                args.epochs,
                args.learning_rate,
                args.patience,
                args.tolerance,
                args.add_sigmoid,
                'Adam',
                0.1,
                args.use_gpu,
                args.batch_size,
                args.evaluation_steps,
                args.nproc,
                args.transfer_learning_path,
                args.transfer_learning_autoencoder,
                args.selection)
        train_slice(train_params_slice)
    elif args.mode=='patch':
       if args.train_autoencoder :
           train_params_autoencoder = Parameters(args.tsv_path, 
                   args.output_dir, 
                   args.caps_dir, 
                   args.network)
           train_params_autoencoder.write(args.pretrained_path,
                   args.pretrained_difference,
                   args.preprocessing,
                   args.diagnoses,
                   args.baseline,
                   args.minmaxnormalization,
                   'random',
                   args.n_splits,
                   args.split,
                   args.accumulation_steps,
                   args.epochs,
                   args.learning_rate,
                   args.patience,
                   args.tolerance,
                   args.add_sigmoid,
                   'Adam',
                   0.0,
                   args.use_gpu,
                   args.batch_size,
                   args.evaluation_steps,
                   args.nproc)
           train_autoencoder_patch(train_params_autoencoder)
       else:
           train_params_patch = Parameters(args.tsv_path, 
                   args.output_dir, 
                   args.caps_dir, 
                   args.network)
           train_params_patch.write(args.pretrained_path,
                   args.pretrained_difference,
                   args.preprocessing,
                   args.diagnoses,
                   args.baseline,
                   args.minmaxnormalization,
                   args.sampler,
                   args.n_splits,
                   args.split,
                   args.accumulation_steps,
                   args.epochs,
                   args.learning_rate,
                   args.patience,
                   args.tolerance,
                   args.add_sigmoid,
                   'Adam',
                   0.1,
                   args.use_gpu,
                   args.batch_size,
                   args.evaluation_steps,
                   args.nproc,
                   args.transfer_learning_path,
                   args.transfer_learning_autoencoder,
                   args.selection)
           if args.network_type=='single':
               train_patch_single_cnn(train_params_cnn)
           else:
               train_patch_multi_cnn(train_params_cnn)
    elif args.mode=='svn':
        pass

    else:
        print('Mode not detected in clinicadl')

# Function to dispatch command line options from classify to corresponding
# function
def classify_func(args):
    pass



def parse_command_line():
    parser = argparse.ArgumentParser(prog='clinicadl', 
            description='Clinica Deep Learning.')

    subparser = parser.add_subparsers(title='Task to execute with clinicadl',
            description='''What kind of task do you want to use with clinicadl
            (preprocessing, extract, train, validate, classify).''',
            dest='task', 
            help='Stages/task to execute with clinicadl')
    #subparser_extract = parser.add_subparsers(dest='ext',
    #        help='Extract the data')

    subparser.required = True 

    # Preprocessing 1
    # preprocessing_parser: get command line arguments and options for
    # preprocessing

    preprocessing_parser = subparser.add_parser('preprocessing',
        help='Prepare data for training (needs clinica installed).')
    preprocessing_parser.add_argument('bids_directory',
        help='Data using BIDS structure.',
        default=None)
    preprocessing_parser.add_argument('caps_dir',
        help='Data using CAPS structure.',
        default=None)
    preprocessing_parser.add_argument('tsv_file',
        help='tsv file with sujets/sessions to process.',
        default=None)
    preprocessing_parser.add_argument('ref_template',
        help='Template reference.',
        default=None)
    preprocessing_parser.add_argument('working_directory',
        help='Working directory to save temporary file.',
        default=None)
    preprocessing_parser.add_argument('-np', '--nproc',
        help='Number of cores used for processing (2 by default)',
        type=int, default=2)


    preprocessing_parser.set_defaults(func=preprocessing_t1w_func)

    # Preprocessing 2 - Extract data: slices or patches
    # extract_parser: get command line argument and options

    extract_parser = subparser.add_parser('extract',
        help='Create data (slices or patches) for training.')
    extract_parser.add_argument('caps_dir',
        help='Data using CAPS structure.',
        default=None)
    extract_parser.add_argument('tsv_file',
        help='tsv file with sujets/sessions to process.',
        default=None)
    extract_parser.add_argument('working_directory',
        help='Working directory to save temporary file.',
        default=None)
    extract_parser.add_argument('extract_method',
        help='Method used to extract features: slice or patch',
        choices=['slice', 'patch'], default=None)
    extract_parser.add_argument('-psz', '--patch_size',
        help='Patch size e.g: --patch_size 50',
        type=int, default=50)
    extract_parser.add_argument('-ssz', '--stride_size',
        help='Stride size  e.g.: --stride_size 50',
        type=int, default=50)
    extract_parser.add_argument('-sd', '--slice_direction',
        help='Slice direction',
        type=int, default=0)
    extract_parser.add_argument('-sm', '--slice_mode',
        help='Slice mode',
        choices=['original', 'rgb'], default='rgb')
    extract_parser.add_argument('-np', '--nproc',
        help='Number of cores used for processing',
        type=int, default=2)
    
    extract_parser.set_defaults(func=extract_data_func)
   
    
    # Train - Train CNN model with preprocessed  data
    # train_parser: get command line arguments and options

    train_parser = subparser.add_parser('train',
        help='Train with your data and create a model.')
    train_parser.add_argument('mode',
        help='Choose your mode (subject level, slice level, patch level, svm).',
        choices=['subject', 'slice', 'patch', 'svm'],
        default='subject')
    train_parser.add_argument('caps_dir',
        help='Data using CAPS structure.',
        default=None)
    train_parser.add_argument('tsv_path',
        help='tsv path with sujets/sessions to process.',
        default=None)
    train_parser.add_argument('output_dir',
        help='Folder containing results of the training.',
        default=None)
    train_parser.add_argument('network',
        help='CNN Model to be used during the training',
        default='Conv5_FC3')
    
    ## Optional parameters
    ## Computational issues
    train_parser.add_argument('-gpu', '--use_gpu', action='store_true',
        help='Uses gpu instead of cpu if cuda is available',
        default=False)
    train_parser.add_argument('-np', '--nproc',
        help='Number of cores used during the training',
        type=int, default=2)
    train_parser.add_argument("--batch_size", 
        default=2, type=int,
        help='Batch size for training. (default=2)',)
    train_parser.add_argument('--evaluation_steps', '-esteps', 
        default=1, type=int,
        help='Fix the number of batches to use before validation')

    ## Data Management
    train_parser.add_argument('--preprocessing',
        help='Defines the type of preprocessing of CAPS data.',
        choices=['linear', 'mni'], type=str,
        default='linear')
    train_parser.add_argument('--diagnoses', '-d',
        help='Take all the subjects possible for autoencoder training',
        default=['AD', 'CN'], nargs='+', type=str)
    train_parser.add_argument('--baseline',
        help='if True only the baseline is used',
        action="store_true",
        default=False)
    train_parser.add_argument('--minmaxnormalization', '-n', 
        help='Performs MinMaxNormalization',
        action="store_true",
        default=False)

    ## Cross-validation
    train_parser.add_argument('--n_splits', 
        help='If a value is given will load data of a k-fold CV', 
        type=int, default=None)
    train_parser.add_argument('--split', 
        help='Will load the specific split wanted.', 
        type=int, default=0)

    ## Training arguments
    train_parser.add_argument('--network_type',
        help='Chose between sinlge or multi CNN (applies only for mode patch-level)',
        choices=['single', 'multi'], type=str,
        default='single')
    train_parser.add_argument('-tAE', '--train_autoencoder', 
        help='Add this option if you want to train an autoencoder',
        action="store_true",
        default=False)
    train_parser.add_argument('--sampler', '-sm',
        help='Sampler to be used',
        default='random', type=str)
    train_parser.add_argument('--accumulation_steps', '-asteps',
        help='Accumulates gradients in order to increase the size of the batch',
        default=1, type=int)
    train_parser.add_argument('--epochs', 
        help='Epochs through the data. (default=20)',
        default=20, type=int)
    train_parser.add_argument('--learning_rate', '-lr',
        help='Learning rate of the optimization. (default=0.01)',
        default=1e-4, type=float)
    train_parser.add_argument('--patience', 
        help='Waiting time for early stopping.',
        type=int, default=10)
    train_parser.add_argument('--tolerance', 
        help='Tolerance value for the early stopping.',
        type=float, default=0.0)
    train_parser.add_argument('--add_sigmoid', 
        help='Ad sigmoid function at the end of the decoder.',
        default=False, action="store_true")

    ## Transfer learning from other autoencoder/network
    train_parser.add_argument('--pretrained_path', 
        help='Path to a pretrained model (can be of different size).',
        type=str, default=None)
    train_parser.add_argument("--pretrained_difference",
        help='''Difference of size between the pretrained autoencoder and 
            the training one. If the new one is larger, difference will be 
            positive.''',
        type=int, default=0)
    train_parser.add_argument("--transfer_learning_path", 
        help="If an existing path is given, a pretrained autoencoder is used.",
        type=str, default=None)
    train_parser.add_argument("--transfer_learning_autoencoder",
        help="If do transfer learning using an autoencoder else will look for a CNN model.",
        default=False, action="store_true")
    train_parser.add_argument("--selection", 
        help="Allow to choose which model of the experiment is loaded.",
        type=str, default="best_acc", choices=["best_loss", "best_acc"])

    train_parser.set_defaults(func=train_func)
    
    
    # Classify - Classify a subject or a list of tesv files with the CNN
    # provieded as argument.
    # classify_parser: get command line arguments and options

    classify_parser = subparser.add_parser('classify',
        help='Classify one image or a list of images with your previouly trained model.')
    classify_parser.add_argument('mode',
        help='Choose your mode (subject level, slice level, patch level, svm).',
        choices=['subject', 'slice', 'patch', 'svm'],
        default='subject')
    classify_parser.add_argument('caps_dir',
        help='Data using CAPS structure.',
        default=None)
    classify_parser.add_argument('tsv_path',
        help='tsv path with sujets/sessions to process.',
        default=None)
    classify_parser.add_argument('output_dir',
        help='Folder containing results of the training.',
        default=None)
    classify_parser.add_argument('network_dir',
        help='Path to the folder where the model was saved during the training.',
        default=None)

    classify_parser.set_defaults(func=classify_func)
    
    args = parser.parse_args()
    
    commandline = parser.parse_known_args()
    commandline_to_json(commandline, 'model_type')
    
    #print(args)
   
    return args
