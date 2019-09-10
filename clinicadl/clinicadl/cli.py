import argparse

from clinicadl.preprocessing.T1_preprocessing import preprocessing_t1w
from clinicadl.preprocessing.T1_postprocessing import postprocessing_t1w

def preprocessing_t1w_func(args):
    wf = preprocessing_t1w(args.bids_directory, 
            args.caps_directory,
            args.tsv_file,
            args.ref_template,
            args.working_directory)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})

def postprocessing_t1w_func(args):
    wf = postprocessing_t1w(args.caps_directory, 
            args.tsv_file,
            args.patch.size,
            args.stride_size,
            args.working_directory,
            args.extract_method,
            args.slice_direction,
            args.slice_mode)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})

def parse_command_line():
    parser = argparse.ArgumentParser(description='Clinica Deep Learning.')

    subparsers = parser.add_subparsers(dest='cmd', help='subcommands')

    subparsers.required = True

    # Preprocessing 1
    preprocessing1 = subparsers.add_parser('preprocessing',
            help='Prepare data for training')
    preprocessing1.add_argument('-bd', '--bids_directory',
            help='Data using BIDS structure.')
    preprocessing1.add_argument('-cd', '--caps_directory',
            help='Data using CAPS structure.')
    preprocessing1.add_argument('-tsv', '--tsv_file',
            help='tsv file with sujets/sessions to process.')
    preprocessing1.add_argument('-rt', '--ref_template',
            help='Template reference.')
    preprocessing1.add_argument('-wd', '--working_directory', default=None,
            help='Working directory to save temporary file.')


    preprocessing1.set_defaults(func=preprocessing_t1w_func)

    # Preprocessing 2 - Create slices and patches
    
    preprocessing2 = subparsers.add_parser('createdata',
            help='Create data (slices or patches) for training')
    preprocessing2.add_argument('-cd', '--caps_directory',
            help='Data using CAPS structure.')
    preprocessing2.add_argument('-tsv', '--tsv_file',
            help='tsv file with sujets/sessions to process.')
    preprocessing2.add_argument('-wd', '--working_directory', default=None,
            help='Working directory to save temporary file.')
    preprocessing2.add_argument('-psz', '--patch_size',
            help='Patch size')
    preprocessing2.add_argument('-ssz', '--stride_size',
            help='Stride size')
    preprocessing2.add_argument('-ex', '--extract_method',
            help='Method used to extact features: slice or patch')
    preprocessing2.add_argument('-sd', '--slice_direction',
            help='Slice direction')
    preprocessing2.add_argument('-sm', '--slice_mode',
            help='Slice mode')
    
    
    preprocessing2.set_defaults(func=postprocessing_t1w_func)
   
    
    args = parser.parse_args()
    
    return args
