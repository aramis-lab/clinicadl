import argparse

from clinicadl.preprocessing.T1_preprocessing import preprocessing_t1w
from clinicadl.preprocessing.T1_postprocessing import postprocessing_t1w

def preprocessing_t1w_func(args):
    wf = preprocessing_t1w(args.bids_directory, 
            args.caps_directory,
            args.tsv_file,
            args.ref_template,
            args.working_directory)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})

def extract_data_func(args):
    wf = postprocessing_t1w(args.caps_directory, 
            args.tsv_file,
            args.patch_size,
            args.stride_size,
            args.working_directory,
            args.extract_method,
            args.slice_direction,
            args.slice_mode)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})

def train_func(args):
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
    preprocessing_parser.add_argument('caps_directory',
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
    extract_parser.add_argument('caps_directory',
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
    train_parser.add_argument('model',
            help='Choice your model (subject level, slice level, patch level, svm).',
            default=None)
    train_parser.add_argument('caps_directory',
            help='Data using CAPS structure.',
            default=None)
    
    train_parser.set_defaults(func=train_func)
    
    
    args = parser.parse_args()
    
    print(args)
    return args
