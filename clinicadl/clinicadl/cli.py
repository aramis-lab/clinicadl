# coding: utf8

import argparse
from distutils.util import strtobool

from colorama import Fore


TRAIN_CATEGORIES = {
    # General parent group
    'POSITIONAL': '%sPositional arguments%s' % (Fore.BLUE, Fore.RESET),
    'COMPUTATIONAL': '%sComputational resources%s' % (Fore.BLUE, Fore.RESET),
    'DATA': '%sData management%s' % (Fore.BLUE, Fore.RESET),
    'CROSS-VALIDATION': '%sCross-validation arguments%s' % (Fore.BLUE, Fore.RESET),
    'OPTIMIZATION': '%sOptimization parameters%s' % (Fore.BLUE, Fore.RESET),
    # Other parent groups
    'TRANSFER LEARNING': '%sTransfer learning%s' % (Fore.BLUE, Fore.RESET),
    'AUTOENCODER': '%sAutoencoder specific%s' % (Fore.BLUE, Fore.RESET),
    # Slice-level
    'SLICE': '%sSlice-level parameters%s' % (Fore.BLUE, Fore.RESET),
    'SLICE CNN': '%sSlice-level CNN parameters%s' % (Fore.BLUE, Fore.RESET),
    # Patch arguments
    'PATCH': '%sPatch-level parameters%s' % (Fore.BLUE, Fore.RESET),
    'PATCH CNN': '%sPatch-level CNN parameters%s' % (Fore.BLUE, Fore.RESET),
    # ROI-based arguments
    'ROI': '%sROI-based parameters%s' % (Fore.BLUE, Fore.RESET),
    'ROI CNN': '%sROI-based CNN parameters%s' % (Fore.BLUE, Fore.RESET),
    # Other optional arguments
    'OPTIONAL': '%sOther options%s' % (Fore.BLUE, Fore.RESET),
    # Model selection
    'MODEL': '%sModel selection%s' % (Fore.BLUE, Fore.RESET),
    # Display
    'DISPLAY': '%sResults display%s' % (Fore.BLUE, Fore.RESET),
}


def extract_tensors(args):
    import sys
    from clinica.utils.stream import FilterOut
    from clinica.pipelines.deeplearning_prepare_data.deeplearning_prepare_data_cli import DeepLearningPrepareDataCLI

    sys.stdout = FilterOut(sys.stdout)

    dl_prepare_data_cli = DeepLearningPrepareDataCLI()
    dl_prepare_data_cli.run_command(args)


def qc_func(args):
    from .quality_check.t1_linear.quality_check import quality_check as linear_qc
    from .quality_check.t1_volume.quality_check import quality_check as volume_qc

    if args.preprocessing == "t1-linear":
        linear_qc(
            args.caps_dir,
            args.output_path,
            tsv_path=args.subjects_sessions_tsv,
            threshold=args.threshold,
            batch_size=args.batch_size,
            num_workers=args.nproc,
            gpu=not args.use_cpu
        )
    elif args.preprocessing == "t1-volume":
        volume_qc(
            args.caps_dir,
            args.output_dir,
            args.group_label
        )


def generate_data_func(args):
    from .tools.data.generate_data import (generate_random_dataset,
                                           generate_trivial_dataset,
                                           generate_shepplogan_dataset)

    if args.mode == "random":
        generate_random_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.subjects_sessions_tsv,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            mean=args.mean,
            sigma=args.sigma,
            preprocessing=args.preprocessing)
    elif args.mode == "trivial":
        generate_trivial_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.subjects_sessions_tsv,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            preprocessing=args.preprocessing,
            mask_path=args.mask_path,
            atrophy_percent=args.atrophy_percent,
        )
    else:
        labels_distribution = {"AD": args.AD_subtypes_distribution, "CN": args.CN_subtypes_distribution}
        generate_shepplogan_dataset(
            output_dir=args.output_dir,
            img_size=args.image_size,
            labels_distribution=labels_distribution,
            samples=args.n_subjects,
            smoothing=args.smoothing
        )


def rs_func(args):
    from .train.random_search import launch_search

    launch_search(args)


# Function to dispatch training to corresponding function
def train_func(args):
    from .train import train_autoencoder, train_multi_cnn, train_single_cnn

    if args.network_type == "autoencoder":
        args.transfer_learning_selection = "best_loss"
        train_autoencoder(args)
    elif args.network_type == "cnn":
        train_single_cnn(args)
    elif args.network_type == "multicnn":
        train_multi_cnn(args)
    else:
        raise NotImplementedError('Framework %s not implemented in clinicadl' % args.network_type)


# Function to dispatch command line options from classify to corresponding
# function
def classify_func(args):
    from .classify.inference import classify

    classify(
        args.caps_directory,
        args.tsv_path,
        args.model_path,
        args.prefix_output,
        labels=not args.no_labels,
        gpu=not args.use_cpu,
        prepare_dl=args.use_extracted_features,
        selection_metrics=args.selection_metrics,
        diagnoses=args.diagnoses,
        verbose=args.verbose,
        multi_cohort=args.multi_cohort
    )


# Functions to dispatch command line options from tsvtool to corresponding
# function
def tsv_restrict_func(args):
    from .tools.tsv.restriction import aibl_restriction, oasis_restriction

    if args.dataset == "AIBL":
        aibl_restriction(args.merged_tsv, args.results_path)
    elif args.dataset == "OASIS":
        oasis_restriction(args.merged_tsv, args.results_path)


def tsv_getlabels_func(args):
    from .tools.tsv.data_formatting import get_labels

    get_labels(
        args.merged_tsv,
        args.missing_mods,
        args.results_path,
        diagnoses=args.diagnoses,
        modality=args.modality,
        restriction_path=args.restriction_path,
        time_horizon=args.time_horizon,
        variables_of_interest=args.variables_of_interest,
        remove_smc=not args.keep_smc,
        verbose=args.verbose
    )


def tsv_split_func(args):
    from .tools.tsv.data_split import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_test=args.n_test,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        p_age_threshold=args.p_age_threshold,
        p_sex_threshold=args.p_sex_threshold,
        ignore_demographics=args.ignore_demographics,
        verbose=args.verbose,
        categorical_split_variable=args.categorical_split_variable
    )


def tsv_kfold_func(args):
    from .tools.tsv.kfold_split import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_splits=args.n_splits,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        stratification=args.stratification,
        verbose=args.verbose
    )


def tsv_analysis_func(args):
    from .tools.tsv.demographics_analysis import demographics_analysis

    demographics_analysis(
        args.merged_tsv,
        args.formatted_data_path,
        args.results_path,
        diagnoses=args.diagnoses
    )


def interpret_func(args):
    from .interpret.group_backprop import group_backprop
    from .interpret.individual_backprop import individual_backprop

    if args.task == "group":
        group_backprop(args)
    elif args.task == "individual":
        individual_backprop(args)
    else:
        raise ValueError("Unknown task %s for interpretation" % args.task)


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='clinicadl',
        description='Deep learning software for neuroimaging datasets')

    parser.add_argument('-l', '--logname',
                        dest='logname',
                        default="clinicaDL.log",
                        metavar=('file.log'),
                        help='Define the log file name (default: clinicaDL.log)')
    parser.add_argument("-V", "--version",
                        dest='version',
                        action='store_true', default=False,
                        help="ClinicaDL's installed version")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--verbose', '-v', action='count', default=0)

    subparser = parser.add_subparsers(
        title='''Task to execute with clinicadl:''',
        description='''What kind of task do you want to use with clinicadl?
            (tsvtool, preprocessing, generate, train, classify).''',
        dest='task',
        help='''****** Tasks proposed by clinicadl ******''')

    # subparser.required = True

    # Generate synthetic data
    generate_parser = subparser.add_parser(
        'generate',
        help='Generate synthetic data for functional tests.'
    )

    generate_subparser = generate_parser.add_subparsers(
        title='''Type of synthetic data generated''',
        description='''What type of synthetic data do you want to generate?
                (random, shepplogan, trivial).''',
        dest='mode',
        help='''****** Synthetic datasets proposed by clinicadl ******''')

    # generate_subparser.required = True

    # Positional arguments
    generate_rs_parent_parser = argparse.ArgumentParser(add_help=False)
    generate_rs_parent_parser.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None
    )
    generate_rs_parent_parser.add_argument(
        'preprocessing',
        type=str,
        choices=['t1-linear', 't1-extensive'],
        help="Preprocessing used to generate synthetic data."
    )
    generate_rs_parent_parser.add_argument(
        'output_dir',
        help='Folder containing the synthetic dataset.',
    )
    generate_rs_parent_parser.add_argument(
        "--subjects_sessions_tsv", "-tsv",
        help='TSV file containing a list of subjects with their sessions.',
        type=str, default=None
    )
    generate_rs_parent_parser.add_argument(
        '--n_subjects',
        type=int,
        default=300,
        help="Number of subjects in each class of the synthetic dataset."
    )

    generate_random_parser = generate_subparser.add_parser(
        "random",
        parents=[
            parent_parser,
            generate_rs_parent_parser],
        help="Generate a random dataset in which gaussian noise is added to brain images.")
    generate_random_parser.add_argument(
        '--mean',
        type=float,
        default=0,
        help="Mean value of the noise added for the random dataset."
    )
    generate_random_parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help="Standard deviation of the noise added for the random dataset."
    )
    generate_random_parser.set_defaults(func=generate_data_func)

    generate_trivial_parser = generate_subparser.add_parser(
        "trivial",
        parents=[
            parent_parser,
            generate_rs_parent_parser],
        help="Generate a trivial dataset in which gaussian half of the brain is atrophied.")
    generate_trivial_parser.add_argument(
        '--mask_path',
        type=str,
        help='path to the extracted masks to generate the two labels.',
        default=None
    )
    generate_trivial_parser.add_argument(
        '--atrophy_percent',
        type=float,
        default=60,
        help='percentage of atrophy applied'
    )
    generate_trivial_parser.set_defaults(func=generate_data_func)

    generate_shepplogan_parser = generate_subparser.add_parser(
        "shepplogan",
        help="Generate a dataset of 2D images including 3 subtypes based on Shepp Logan phantom."
    )
    generate_shepplogan_parser.add_argument(
        'output_dir',
        help='Folder containing the synthetic dataset.',
    )
    generate_shepplogan_parser.add_argument(
        '--n_subjects',
        type=int,
        default=300,
        help="Number of subjects in each class of the synthetic dataset."
    )
    generate_shepplogan_parser.add_argument(
        '--image_size',
        type=int,
        default=128,
        help="Size in pixels of the squared images."
    )
    generate_shepplogan_parser.add_argument(
        '--CN_subtypes_distribution', '-Csd',
        type=float, nargs='+',
        default=[1.0, 0.0, 0.0],
        help="Probability of each subtype to be drawn in CN label."
    )
    generate_shepplogan_parser.add_argument(
        '--AD_subtypes_distribution', '-Asd',
        type=float, nargs='+',
        default=[0.05, 0.85, 0.10],
        help="Probability of each subtype to be drawn in AD label."
    )
    generate_shepplogan_parser.add_argument(
        '--smoothing',
        action='store_true',
        default=False,
        help='Adds random smoothing to generated data.'
    )
    generate_shepplogan_parser.set_defaults(func=generate_data_func)

    # Preprocessing
    from clinica.pipelines.t1_linear.t1_linear_cli import T1LinearCLI
    from clinica.engine.cmdparser import init_cmdparser_objects
    from .preprocessing.t1_extensive.t1_extensive_cli import T1ExtensiveCli
    preprocessing_parser = subparser.add_parser(
        'preprocessing',
        help='Preprocess T1w-weighted images with t1-linear or t1-extensive pipelines.'
    )

    preprocessing_subparser = preprocessing_parser.add_subparsers(
        title='''Preprocessing task to execute with clinicadl:''',
        description='''What kind of task do you want to perform with clinicadl?
                (run, quality-check, extract-tensor).''',
        dest='preprocessing_task',
        help='''****** Tasks proposed by clinicadl ******''')
    preprocessing_subparser.required = True

    run_parser = preprocessing_subparser.add_parser(
        'run',
        help='Preprocess T1w-weighted images with t1-linear or t1-extensive pipelines.'
    )
    run_parser._positionals.title = ('%sclinicadl preprocessing expects one of the following pipelines%s'
                                     % (Fore.GREEN, Fore.RESET))

    def preprocessing_help(args):
        print('%sNo pipeline was specified. Type clinica preprocessing -h for details%s' %
              (Fore.RED, Fore.RESET))

    run_parser.set_defaults(func=preprocessing_help)

    init_cmdparser_objects(
        parser,
        run_parser.add_subparsers(dest='preprocessing'),
        [
            T1LinearCLI(),
            T1ExtensiveCli(),
        ]
    )

    extract_parser = preprocessing_subparser.add_parser(
        'extract-tensor',
        help='Prepare data generated Clinica for PyTorch with Tensor extraction (image, patches or slices).'
    )
    clinica_comp = extract_parser.add_argument_group('%sClinica mandatory arguments%s' % (Fore.BLUE, Fore.RESET))
    clinica_comp.add_argument(
        "caps_directory",
        help='Path to the CAPS directory.'
    )
    clinica_comp.add_argument("modality",
                              help='''For which modality the tensor will be extracted.
            't1-linear': images prepocessed with t1-linear pipeline.
            't1-extensive': images preprocessed with t1-extensive pipeline.
            'custom': find images with a custom suffix in their filename and
            transform them to tensor format.''',
                              choices=['t1-linear', 't1-extensive', 'custom'], default='t1-linear'
                              )

    clinica_comp.add_argument(
        "extract_method",
        help='''Format of the extracted features. Three options:
        'image' to convert to PyTorch tensor the complete 3D image,
        'patch' to extract 3D volumetric patches and
        'slice' to extract 2D slices from the image.
        By default the features are extracted from the cropped image.''',
        choices=['image', 'slice', 'patch'],
        default='image'
    )

    optional = extract_parser.add_argument_group('%sPipeline options%s' % (Fore.BLUE, Fore.RESET))
    optional.add_argument(
        '-uui', '--use_uncropped_image',
        help='''Use the uncropped image instead of the
        cropped image generated by t1-linear.''',
        default=False, action="store_true"
    )

    optional_patch = extract_parser.add_argument_group(
        "%sPipeline options if you chose ‘patch’ extraction%s" % (Fore.BLUE, Fore.RESET)
    )
    optional_patch.add_argument(
        '-ps', '--patch_size',
        help='''Patch size (default: --patch_size 50).''',
        type=int, default=50
    )
    optional_patch.add_argument(
        '-ss', '--stride_size',
        help='''Stride size (default: --stride_size 50).''',
        type=int, default=50
    )
    optional_slice = extract_parser.add_argument_group(
        "%sPipeline options if you chose ‘slice’ extraction%s" % (Fore.BLUE, Fore.RESET)
    )
    optional_slice.add_argument(
        '-sd', '--slice_direction',
        help='''Slice direction. Three options:
            '0' -> Sagittal plane,
            '1' -> Coronal plane or
            '2' -> Axial plane
            (default: sagittal plane i.e. --slice_direction 0)''',
        type=int, default=0
    )
    optional_slice.add_argument(
        '-sm', '--slice_mode',
        help='''Slice mode. Two options: 'rgb' to save the slice in
            three identical channels, ‘single’ to save the slice in a
            single channel (default: --slice_mode rgb).''',
        choices=['rgb', 'single'], default='rgb'
    )
    optional_custom = extract_parser.add_argument_group(
        "%sPipeline options if you chose ‘custom’ modality%s" % (Fore.BLUE, Fore.RESET)
    )
    optional_custom.add_argument(
        '-cn', '--custom_suffix',
        help='''Custom suffix filename, e.g.:
            'graymatter_space-Ixi549Space_modulated-off_probability.nii.gz', or
            'segm-whitematter_probability.nii.gz'
            ''',
        type=str, default=''
    )

    # Clinica standard arguments (e.g. --n_procs)
    clinica_standard_options = extract_parser.add_argument_group('%sClinica standard options%s' % (Fore.BLUE, Fore.RESET))
    clinica_standard_options.add_argument(
        "-tsv", "--subjects_sessions_tsv",
        help='TSV file containing a list of subjects with their sessions.'
    )
    clinica_standard_options.add_argument(
        "-wd", "--working_directory",
        help='Temporary directory to store pipelines intermediate results.'
    )
    clinica_standard_options.add_argument(
        "-np", "--n_procs",
        metavar='N', type=int,
        help='Number of cores used to run in parallel.'
    )

    extract_parser.set_defaults(func=extract_tensors)

    qc_parser = preprocessing_subparser.add_parser(
        'quality-check',
        help='Performs quality check procedure for t1-linear pipeline.'
             'Original code can be found at https://github.com/vfonov/deep-qc'
    )

    qc_subparsers = qc_parser.add_subparsers(
        title='''Preprocessing pipelines available''',
        description='''Which preprocessing pipeline do you want to check?''',
        dest='preprocessing',
        help='''****** Preprocessing pipelines ******''')

    qc_subparsers.required = True
    qc_linear_parser = qc_subparsers.add_parser(
        't1-linear',
        help='Performs quality check on t1-linear pipeline.'
    )
    qc_linear_parser.add_argument("caps_dir",
                                  help='Data using CAPS structure.',
                                  type=str)
    qc_linear_parser.add_argument("output_path",
                                  help="Path to the output tsv file (filename included).",
                                  type=str)

    qc_linear_parser.add_argument("--subjects_sessions_tsv", "-tsv",
                                  help='TSV file containing a list of subjects with their sessions.',
                                  type=str, default=None)
    qc_linear_parser.add_argument("--threshold",
                                  help='The threshold on the output probability to decide if the image '
                                       'passed or failed. (default=0.5)',
                                  type=float, default=0.5)
    qc_linear_parser.add_argument('--batch_size',
                                  help='Batch size used in DataLoader (default=1).',
                                  default=1, type=int)
    qc_linear_parser.add_argument("-np", "--nproc",
                                  help='Number of cores used the quality check. (default=2)',
                                  type=int, default=2)
    qc_linear_parser.add_argument('-cpu', '--use_cpu', action='store_true',
                                  help='If provided, will use CPU instead of GPU.',
                                  default=False)
    qc_linear_parser.set_defaults(func=qc_func)

    qc_volume_parser = qc_subparsers.add_parser(
        't1-volume',
        help='Performs quality check on t1-volume pipeline.'
    )
    qc_volume_parser.add_argument("caps_dir",
                                  help='Data using CAPS structure.',
                                  type=str)
    qc_volume_parser.add_argument("output_dir",
                                  help="Path to the output directory containing TSV files.",
                                  type=str)
    qc_volume_parser.add_argument("group_label",
                                  help="Identifier for the group of subjects used to create the DARTEL template.",
                                  type=str)
    qc_volume_parser.set_defaults(func=qc_func)

    # random search parsers
    rs_generate_parser = subparser.add_parser(
        'random-search',
        parents=[parent_parser],
        help='Generate random networks to explore hyper parameters space.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # rs_subparsers = rs_parser.add_subparsers(
    #     title='''Possibilities for random network training''',
    #     description='''You can generate and train a new random network,
    #     or relaunch a previous random job with some alterations.''',
    #     dest='random_task',
    #     help='''****** Possible tasks ******'''
    # )
    # rs_subparsers.required = True

    # rs_generate_parser = rs_subparsers.add_parser(
    #     'generate',
    #     parents=[parent_parser],
    #     help='Sample a new network and train it.',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    rs_pos_group = rs_generate_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"]
    )
    rs_pos_group.add_argument("launch_dir", type=str,
                              help="Directory containing the random_search.json file.")
    rs_pos_group.add_argument("name", type=str,
                              help="Name of the job.")

    # Data management
    rs_data_group = rs_generate_parser.add_argument_group(
        TRAIN_CATEGORIES["CROSS-VALIDATION"]
    )
    rs_data_group.add_argument("--n_splits", type=int, default=None,
                               help="If a value is given will load data of a k-fold CV")
    rs_data_group.add_argument("--split", type=int, default=None, nargs="+",
                               help="Will load the specific split wanted.")

    rs_comp_group = rs_generate_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"]
    )
    rs_comp_group.add_argument(
        '-cpu', '--use_cpu', action='store_true',
        help='If provided, will use CPU instead of GPU.',
        default=False)
    rs_comp_group.add_argument(
        '-np', '--nproc',
        help='Number of cores used during the training.',
        type=int, default=2)
    rs_comp_group.add_argument(
        '--batch_size',
        default=2, type=int,
        help='Batch size for training.')
    rs_comp_group.add_argument(
        '--evaluation_steps', '-esteps',
        default=0, type=int,
        help='Fix the number of iterations to perform before computing an evaluations. Default will only '
             'perform one evaluation at the end of each epoch.')

    rs_generate_parser.set_defaults(func=rs_func)

    # retrain_parent_parser = return_train_parent_parser(retrain=True)
    # rs_retrain_parser = rs_subparsers.add_parser(
    #     'retrain',
    #     parents=[parent_parser, retrain_parent_parser],
    #     help='Train a network previously created by generate.',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )

    train_parser = subparser.add_parser(
        'train',
        help='Train with your data and create a model.')

    train_subparser = train_parser.add_subparsers(
        title='''Inputs types implemented in clinicadl''',
        description='''What type of input do you want to use?
                (image, patch, roi, slice).''',
        dest='mode',
        help='''****** Input types proposed by clinicadl ******''')

    train_subparser.required = True

    # Transfer learning
    transfer_learning_parent = argparse.ArgumentParser(add_help=False)
    transfer_learning_group = transfer_learning_parent.add_argument_group(
        TRAIN_CATEGORIES["TRANSFER LEARNING"])
    transfer_learning_group.add_argument(
        '--transfer_learning_path',
        help="If an existing path is given, a pretrained model is used.",
        type=str, default=None)

    # Autoencoder
    autoencoder_parent = argparse.ArgumentParser(add_help=False)
    autoencoder_group = autoencoder_parent.add_argument_group(
        TRAIN_CATEGORIES["AUTOENCODER"])
    autoencoder_group.add_argument(
        '--visualization',
        help='Save examples of image reconstructions.',
        action="store_true",
        default=False)

    ######################
    # IMAGE
    ######################
    train_image_parser = train_subparser.add_parser(
        "image",
        help="Train a 3D image-level network.")

    train_image_subparser = train_image_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or cnn classification ?''',
        dest='network_type',
        help='''****** Choose a type of network ******''')

    train_parent_parser = return_train_parent_parser(retrain=False)
    train_image_ae_parser = train_image_subparser.add_parser(
        "autoencoder",
        parents=[
            parent_parser,
            train_parent_parser,
            autoencoder_parent,
            transfer_learning_parent],
        help="Train an image-level autoencoder.")

    train_image_ae_parser.set_defaults(func=train_func)

    train_image_cnn_parser = train_image_subparser.add_parser(
        "cnn",
        parents=[
            parent_parser,
            train_parent_parser,
            transfer_learning_parent],
        help="Train an image-level CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_image_cnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_image_cnn_parser.set_defaults(func=train_func)

    #########################
    # PATCH
    #########################
    train_patch_parser = train_subparser.add_parser(
        "patch",
        help="Train a 3D patch-level network.")

    train_patch_parent = argparse.ArgumentParser(add_help=False)
    train_patch_group = train_patch_parent.add_argument_group(
        TRAIN_CATEGORIES["PATCH"])
    train_patch_group.add_argument(
        '-ps', '--patch_size',
        help='Patch size',
        type=int, default=50)
    train_patch_group.add_argument(
        '-ss', '--stride_size',
        help='Stride size',
        type=int, default=50)
    train_patch_group.add_argument(
        '--use_extracted_patches',
        help='''If provided the outputs of extract preprocessing are used, else the whole
             MRI is loaded.''',
        default=False, action="store_true")

    train_patch_subparser = train_patch_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or (multi)cnn classification ?''',
        dest='network_type',
        help='''****** Choose a type of network ******''')
    train_patch_subparser.required = True

    train_patch_ae_parser = train_patch_subparser.add_parser(
        "autoencoder",
        parents=[parent_parser, train_parent_parser, train_patch_parent, autoencoder_parent, transfer_learning_parent],
        help="Train a 3D patch-level autoencoder.")

    train_patch_ae_parser.set_defaults(func=train_func)

    train_patch_cnn_parser = train_patch_subparser.add_parser(
        "cnn",
        parents=[
            parent_parser,
            train_parent_parser,
            train_patch_parent,
            transfer_learning_parent],
        help="Train a 3D patch-level CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_patch_cnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_patch_cnn_group = train_patch_cnn_parser.add_argument_group(
        TRAIN_CATEGORIES["PATCH CNN"])
    train_patch_cnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
             subject-level performance. Patches are selected if their balanced
             accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_patch_cnn_parser.set_defaults(func=train_func)

    train_patch_multicnn_parser = train_patch_subparser.add_parser(
        "multicnn",
        parents=[
            parent_parser,
            train_parent_parser,
            train_patch_parent,
            transfer_learning_parent],
        help="Train a 3D patch-level multi-CNN (one CNN is trained per patch location).")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_patch_multicnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_patch_multicnn_group = train_patch_multicnn_parser.add_argument_group(
        TRAIN_CATEGORIES["PATCH CNN"])
    train_patch_multicnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
                 subject-level performance. Patches are selected if their balanced
                 accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_patch_multicnn_parser.set_defaults(func=train_func)

    #########################
    # ROI
    #########################
    train_roi_parser = train_subparser.add_parser(
        "roi",
        help="Train a ROI-based level network.")

    train_roi_parent = argparse.ArgumentParser(add_help=False)
    train_roi_group = train_roi_parent.add_argument_group(
        TRAIN_CATEGORIES["ROI"])
    train_roi_group.add_argument(
        '-rl', '--roi_list',
        help='Names of the regions used for the classification task.'
             'Default will use the hippocampi as described in (Wen et al, 2019).',
        type=str, nargs="+", default=None)
    train_roi_group.add_argument(
        '--uncropped_roi',
        help='If given the image is as large as the whole image. Default will crop the image'
             'with the smallest bounding box possible.',
        action='store_true', default=False)
    train_roi_group.add_argument(
        '--use_extracted_roi',
        help='''If provided the outputs of extract preprocessing are used, else the whole
                 MRI is loaded.''',
        default=False, action="store_true")

    train_roi_subparser = train_roi_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or cnn classification ?''',
        dest='network_type',
        help='''****** Choose a type of network ******''')
    train_roi_subparser.required = True

    train_roi_ae_parser = train_roi_subparser.add_parser(
        "autoencoder",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            autoencoder_parent,
            transfer_learning_parent
        ],
        help="Train a ROI-based autoencoder.")

    train_roi_ae_parser.set_defaults(func=train_func)

    train_roi_cnn_parser = train_roi_subparser.add_parser(
        "cnn",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            transfer_learning_parent],
        help="Train a ROI-based CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_roi_cnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_roi_cnn_group = train_roi_cnn_parser.add_argument_group(
        TRAIN_CATEGORIES["ROI CNN"])
    train_roi_cnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
             subject-level performance. ROIs are selected if their balanced
             accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_roi_cnn_parser.set_defaults(func=train_func)

    train_roi_multicnn_parser = train_roi_subparser.add_parser(
        "multicnn",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            transfer_learning_parent],
        help="Train a ROI-based multi-CNN (one CNN is trained per patch location).")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_roi_multicnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_roi_multicnn_group = train_roi_multicnn_parser.add_argument_group(
        TRAIN_CATEGORIES["ROI CNN"])
    train_roi_multicnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
                     subject-level performance. Patches are selected if their balanced
                     accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_roi_multicnn_parser.set_defaults(func=train_func)

    #########################
    # SLICE
    #########################
    train_slice_parser = train_subparser.add_parser(
        "slice",
        help="Train a 2D slice-level CNN.")

    train_slice_subparser = train_slice_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or cnn classification ?''',
        dest='network_type',
        help='''****** Choose a type of network ******''')
    train_slice_subparser.required = True

    train_slice_parent = argparse.ArgumentParser(add_help=False)
    train_slice_group = train_slice_parent.add_argument_group(
        TRAIN_CATEGORIES["SLICE"])
    train_slice_group.add_argument(
        '--slice_direction', '-sd',
        help='''Which coordinate axis to take for slicing the MRI.
                 0 for sagittal
                 1 for coronal
                 2 for axial direction.''',
        default=0, type=int)
    train_slice_group.add_argument(
        '--discarded_slices',
        help='''Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides.''',
        default=20, type=int, nargs='+'
    )
    train_slice_group.add_argument(
        '--use_extracted_slices',
        help='''If provided the outputs of extract preprocessing are used, else the whole
                 MRI is loaded.''',
        default=False, action="store_true")

    train_slice_ae_parser = train_slice_subparser.add_parser(
        "autoencoder",
        parents=[parent_parser, train_parent_parser, train_slice_parent, transfer_learning_parent],
        help="Train a 2D slice-level autoencoder.")

    train_slice_ae_parser.set_defaults(func=train_func)

    train_slice_cnn_parser = train_slice_subparser.add_parser(
        "cnn",
        parents=[parent_parser, train_parent_parser, train_slice_parent, transfer_learning_parent],
        help="Train a 2D slice-level CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_slice_cnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_slice_cnn_group = train_slice_cnn_parser.add_argument_group(
        TRAIN_CATEGORIES["SLICE CNN"])
    train_slice_cnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
             subject-level performance. Slices are selected if their balanced
             accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_slice_cnn_parser.set_defaults(func=train_func)

    train_slice_multicnn_parser = train_slice_subparser.add_parser(
        "multicnn",
        parents=[parent_parser, train_parent_parser, train_slice_parent, transfer_learning_parent],
        help="Train a 2D slice-level multi-CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_slice_multicnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_slice_multicnn_group = train_slice_multicnn_parser.add_argument_group(
        TRAIN_CATEGORIES["SLICE CNN"])
    train_slice_multicnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
                 subject-level performance. Slices are selected if their balanced
                 accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_slice_multicnn_parser.set_defaults(func=train_func)

    # Classify - Classify a subject or a list of tsv files with the CNN
    # provided as argument.
    # classify_parser: get command line arguments and options

    classify_parser = subparser.add_parser(
        'classify',
        parents=[parent_parser],
        help='''Classify one image or a list of images with your previously
                 trained model.''')
    classify_pos_group = classify_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"])
    classify_pos_group.add_argument(
        'caps_directory',
        help='Data using CAPS structure.',
        default=None)
    classify_pos_group.add_argument(
        'tsv_path',
        help='''Path to the file with subjects/sessions to process.
        If it includes the filename will load the tsv file directly.
        Else will load the baseline tsv files of wanted diagnoses produced by tsvtool.''',
        default=None)
    classify_pos_group.add_argument(
        'model_path',
        help='''Path to the folder where the model is stored. Folder structure
                should be the same obtained during the training.''',
        default=None)
    classify_pos_group.add_argument(
        'prefix_output',
        help='Prefix to name the files resulting from the classify task.',
        type=str)

    # Computational resources
    classify_comput_group = classify_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"])
    classify_comput_group.add_argument(
        '-cpu', '--use_cpu', action='store_true',
        help='Uses CPU instead of GPU.',
        default=False)
    classify_comput_group.add_argument(
        '-np', '--nproc',
        help='Number of cores used during the task.',
        type=int, default=2)
    classify_comput_group.add_argument(
        '--batch_size',
        default=2, type=int,
        help='Batch size for data loading. (default=2)')

    # Specific classification arguments
    classify_specific_group = classify_parser.add_argument_group(
        TRAIN_CATEGORIES["OPTIONAL"]
    )
    classify_specific_group.add_argument(
        '-nl', '--no_labels', action='store_true',
        help='Add this flag if your dataset does not contain a ground truth.',
        default=False)
    classify_specific_group.add_argument(
        '--use_extracted_features',
        help='''If True the extract slices or patche are used, otherwise the they
                will be extracted on the fly (if necessary).''',
        default=False, action="store_true")
    classify_specific_group.add_argument(
        '--selection_metrics',
        help='''List of metrics to find the best models to evaluate. Default will
        classify best model based on balanced accuracy.''',
        choices=['loss', 'balanced_accuracy'],
        default=['balanced_accuracy'],
        nargs='+'
    )
    classify_specific_group.add_argument(
        "--diagnoses",
        help="List of participants that will be classified.",
        nargs="+", type=str, choices=['AD', 'CN', 'MCI', 'sMCI', 'pMCI'], default=None)
    classify_specific_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort classification. In this case, caps_dir and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False
    )

    classify_parser.set_defaults(func=classify_func)

    tsv_parser = subparser.add_parser(
        'tsvtool',
        help='''Handle tsv files for metadata processing and data splits.''')

    tsv_subparser = tsv_parser.add_subparsers(
        title='''Task to execute with tsv tool:''',
        description='''What kind of task do you want to use with tsv tool?
                (restrict, getlabels, split, kfold, analysis).''',
        dest='tsv_task',
        help='''****** Tasks proposed by clinicadl tsv tool ******''')

    tsv_subparser.required = True

    tsv_restrict_subparser = tsv_subparser.add_parser(
        'restrict',
        help='Reproduce restrictions applied to AIBL and OASIS datasets')

    tsv_restrict_subparser.add_argument(
        "dataset",
        help="dataset on which the restriction is performed.",
        choices=["AIBL", "OASIS"],
        type=str)

    tsv_restrict_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_restrict_subparser.add_argument(
        "results_path",
        help="Path to the output tsv file (filename included).",
        type=str)

    tsv_restrict_subparser.set_defaults(func=tsv_restrict_func)

    tsv_getlabels_subparser = tsv_subparser.add_parser(
        'getlabels',
        parents=[parent_parser],
        help='Get labels in separate tsv files.')

    tsv_getlabels_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_getlabels_subparser.add_argument(
        "missing_mods",
        help="Path to the folder where the outputs of clinica iotools missing-mods are.",
        type=str)
    tsv_getlabels_subparser.add_argument(
        "results_path",
        type=str,
        help="Path to the folder where tsv files are extracted.")

    # Optional arguments
    tsv_getlabels_subparser.add_argument(
        "--modality", "-mod",
        help="Modality to select sessions. Sessions which do not include the modality will be excluded.",
        default="t1w", type=str)
    tsv_getlabels_subparser.add_argument(
        "--diagnoses",
        help="Labels that must be extracted from merged_tsv.",
        nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])
    tsv_getlabels_subparser.add_argument(
        "--time_horizon",
        help="Time horizon to analyse stability of MCI subjects.",
        default=36, type=int)
    tsv_getlabels_subparser.add_argument(
        "--restriction_path",
        help="Path to a tsv containing the sessions that can be included.",
        type=str, default=None)
    tsv_getlabels_subparser.add_argument(
        "--variables_of_interest",
        help="Variables of interest that will be kept in the final lists."
             "Default will keep the diagnosis, age and the sex needed for the split procedure.",
        type=str, nargs="+", default=None)
    tsv_getlabels_subparser.add_argument(
        "--keep_smc",
        help="This flag allows to keep SMC participants, else they are removed.",
        default=False, action="store_true"
    )

    tsv_getlabels_subparser.set_defaults(func=tsv_getlabels_func)

    tsv_split_subparser = tsv_subparser.add_parser(
        'split',
        parents=[parent_parser],
        help='Performs one stratified shuffle split on participant level.')

    tsv_split_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)

    # Optional arguments
    tsv_split_subparser.add_argument(
        "--n_test",
        help="If >= 1, number of subjects to put in set with name 'subset_name'. "
             "If < 1, proportion of subjects to put set with name 'subset_name'. "
             "If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name.",
        type=float, default=100.)
    tsv_split_subparser.add_argument(
        "--MCI_sub_categories",
        help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
        action="store_false", default=True)
    tsv_split_subparser.add_argument(
        "--p_sex_threshold", "-ps",
        help="The threshold used for the chi2 test on sex distributions.",
        default=0.80, type=float)
    tsv_split_subparser.add_argument(
        "--p_age_threshold", "-pa",
        help="The threshold used for the T-test on age distributions.",
        default=0.80, type=float)
    tsv_split_subparser.add_argument(
        "--subset_name",
        help="Name of the subset that is complementary to train.",
        type=str, default="test")
    tsv_split_subparser.add_argument(
        "--ignore_demographics",
        help="If True do not use age and sex to create the splits.",
        default=False, action="store_true"
    )
    tsv_split_subparser.add_argument(
        "--categorical_split_variable",
        help="Name of a categorical variable used for a stratified shuffle split "
             "(in addition to age and sex selection).",
        default=None, type=str
    )

    tsv_split_subparser.set_defaults(func=tsv_split_func)

    tsv_kfold_subparser = tsv_subparser.add_parser(
        'kfold',
        parents=[parent_parser],
        help='Performs a k-fold split on participant level.')

    tsv_kfold_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)

    # Optional arguments
    tsv_kfold_subparser.add_argument(
        "--n_splits",
        help="Number of folds in the k-fold split."
             "If 0, there is no training set and the whole dataset is considered as a test set.",
        type=int, default=5)
    tsv_kfold_subparser.add_argument(
        "--MCI_sub_categories",
        help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
        action="store_false", default=True)
    tsv_kfold_subparser.add_argument(
        "--subset_name",
        help="Name of the subset that is complementary to train.",
        type=str, default="validation")
    tsv_kfold_subparser.add_argument(
        "--stratification",
        help="Name of a variable used to stratify the k-fold split.",
        type=str, default=None)

    tsv_kfold_subparser.set_defaults(func=tsv_kfold_func)

    tsv_analysis_subparser = tsv_subparser.add_parser(
        'analysis',
        help='Produces a demographic analysis of the extracted labels.')

    tsv_analysis_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
    tsv_analysis_subparser.add_argument(
        "formatted_data_path",
        help="Path to the folder containing data extracted by clinicadl tsvtool getlabels.",
        type=str)
    tsv_analysis_subparser.add_argument(
        "results_path",
        help="Path to the output tsv file (filename included).",
        type=str)

    # Modality selection
    tsv_analysis_subparser.add_argument(
        "--diagnoses",
        help="Labels selected for the demographic analysis.",
        default=['AD', 'CN'], nargs="+", type=str, choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'])

    tsv_analysis_subparser.set_defaults(func=tsv_analysis_func)

    interpret_parser = subparser.add_parser(
        'interpret',
        help='''Interpret classification performed by a CNN with saliency maps.''')

    interpret_subparser = interpret_parser.add_subparsers(
        title='''Type of saliency map to perform:''',
        description='''Do you want to perform a group saliency map or individual ones?''',
        dest='task',
        help='''****** Saliency maps proposed by clinicadl ******'''
    )
    interpret_subparser.required = True

    interpret_parent_parser = argparse.ArgumentParser(add_help=False)

    interpret_pos_group = interpret_parent_parser.add_argument_group(TRAIN_CATEGORIES["POSITIONAL"])
    interpret_pos_group.add_argument(
        "model_path", type=str,
        help="Path to the model output directory.")
    interpret_pos_group.add_argument(
        "name", type=str,
        help="Name of the interpretation map.")

    interpret_comput_group = interpret_parent_parser.add_argument_group(TRAIN_CATEGORIES["COMPUTATIONAL"])
    interpret_comput_group.add_argument(
        "--batch_size", default=1, type=int,
        help="Batch size for selection of images (keep_true).")
    interpret_comput_group.add_argument(
        '-cpu', '--use_cpu',
        action='store_true', default=False,
        help='Uses gpu instead of cpu if cuda is available.')
    interpret_comput_group.add_argument(
        '-np', '--nproc',
        default=2, type=int,
        help='the number of batches being loaded in parallel.')

    interpret_model_group = interpret_parent_parser.add_argument_group(TRAIN_CATEGORIES["MODEL"])
    interpret_model_group.add_argument(
        "--selection", default=['best_loss'], type=str, nargs="+",
        choices=['best_loss', 'best_balanced_accuracy'],
        help="Loads the model selected on minimal loss or maximum accuracy on validation.")

    interpret_data_group = interpret_parent_parser.add_argument_group(TRAIN_CATEGORIES["DATA"])
    interpret_data_group.add_argument(
        "--tsv_path", type=str, default=None,
        help="TSV path with subjects/sessions to process, if different from classification task.")
    interpret_data_group.add_argument(
        "--caps_dir", type=str, default=None,
        help="Path to input dir of the MRI (preprocessed CAPS_dir), if different from classification task")
    interpret_data_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort interpretation. In this case, caps_dir and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False
    )
    interpret_data_group.add_argument(
        "--diagnosis", "-d", default='AD', type=str,
        help="The images corresponding to this diagnosis only will be loaded.")
    interpret_data_group.add_argument(
        "--target_diagnosis", default=None, type=str,
        help="Which class the gradients explain. If None is given will be equal to diagnosis.")
    interpret_data_group.add_argument(
        "--baseline", action="store_true", default=False,
        help="If provided, only the baseline sessions are used for training.")
    interpret_data_group.add_argument(
        "--keep_true", type=lambda x: bool(strtobool(x)), default=None,
        help="Chooses false or true positive values of the classification. No selection by default")
    interpret_data_group.add_argument(
        "--nifti_template_path", type=str, default=None,
        help="Path to a nifti template to retrieve affine values.")

    interpret_display_group = interpret_parent_parser.add_argument_group(TRAIN_CATEGORIES["DISPLAY"])
    interpret_display_group.add_argument(
        "--vmax", type=float, default=0.5,
        help="Maximum value used in 2D image display.")

    interpret_group_parser = interpret_subparser.add_parser(
        "group",
        parents=[parent_parser, interpret_parent_parser],
        help="Mean saliency map over a list of sessions"
    )

    interpret_group_parser.set_defaults(func=interpret_func)

    interpret_individual_parser = interpret_subparser.add_parser(
        "individual",
        parents=[parent_parser, interpret_parent_parser],
        help="Individual saliency maps for each session in the input TSV file."
    )

    interpret_individual_parser.set_defaults(func=interpret_func)

    return parser


def return_train_parent_parser(retrain=False):
    # Main train parent parser common to train and random search
    train_parent_parser = argparse.ArgumentParser(add_help=False)
    train_pos_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"])
    if retrain:
        train_pos_group.add_argument(
            "model_path", type=str,
            help="Path to the trained model folder.")
        train_pos_group.add_argument(
            "output_dir", type=str,
            help="Directory in which the new job is stored.")
    else:
        train_pos_group.add_argument(
            'caps_dir',
            help='Data using CAPS structure.',
            default=None)
        train_pos_group.add_argument(
            'preprocessing',
            help='Defines the type of preprocessing of CAPS data.',
            choices=['t1-linear', 't1-extensive', 't1-volume'], type=str)
        train_pos_group.add_argument(
            'tsv_path',
            help='TSV path with subjects/sessions to process.',
            default=None)
        train_pos_group.add_argument(
            'output_dir',
            help='Folder containing results of the training.',
            default=None)
        train_pos_group.add_argument(
            'model',
            help='CNN Model to be used during the training.',
            default='Conv5_FC3')

    train_comput_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"])
    train_comput_group.add_argument(
        '-cpu', '--use_cpu', action='store_true',
        help='If provided, will use CPU instead of GPU.',
        default=False)
    train_comput_group.add_argument(
        '-np', '--nproc',
        help='Number of cores used during the training. (default=2)',
        type=int, default=2)
    train_comput_group.add_argument(
        '--batch_size',
        default=2, type=int,
        help='Batch size for training. (default=2)')
    train_comput_group.add_argument(
        '--evaluation_steps', '-esteps',
        default=0, type=int,
        help='Fix the number of iterations to perform before computing an evaluations. Default will only '
             'perform one evaluation at the end of each epoch.')

    train_data_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["DATA"])

    if retrain:
        train_data_group.add_argument(
            "--caps_dir", type=str, default=None,
            help="Data using CAPS structure.")
        train_data_group.add_argument(
            "--tsv_path", type=str, default=None,
            help="TSV path with subjects/sessions to process.")

    train_data_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
        action="store_true",
        default=None if retrain else False
    )
    train_data_group.add_argument(
        '--diagnoses', '-d',
        help='List of diagnoses that will be selected for training.',
        default=None if retrain else ['AD', 'CN'], nargs='+', type=str,
        choices=['AD', 'BV', 'CN', 'MCI', 'sMCI', 'pMCI'])
    train_data_group.add_argument(
        '--baseline',
        help='If provided, only the baseline sessions are used for training.',
        action="store_true",
        default=None if retrain else False)
    train_data_group.add_argument(
        '--unnormalize', '-un',
        help='Disable default MinMaxNormalization.',
        action="store_true",
        default=None if retrain else False)
    train_data_group.add_argument(
        "--data_augmentation", nargs="+", default=None if retrain else False,
        choices=["None", "Noise", "Erasing", "CropPad", "Smoothing"],
        help="Randomly applies transforms on the training set.")
    train_data_group.add_argument(
        '--sampler', '-s',
        help="Sampler choice (random, or weighted for imbalanced datasets)",
        default="random", type=str, choices=["random", "weighted"])

    train_cv_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["CROSS-VALIDATION"])
    train_cv_group.add_argument(
        '--n_splits',
        help='If a value is given will load data of a k-fold CV. Else will load a single split.',
        type=int, default=None)
    train_cv_group.add_argument(
        '--split',
        help='Train the list of given folds. By default train all folds.',
        type=int, default=None, nargs='+')

    train_optim_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["OPTIMIZATION"])
    train_optim_group.add_argument(
        '--epochs',
        help='Maximum number of epochs.',
        default=None if retrain else 20, type=int)
    train_optim_group.add_argument(
        '--learning_rate', '-lr',
        help='Learning rate of the optimization.',
        default=None if retrain else 1e-4, type=float)
    train_optim_group.add_argument(
        '--weight_decay', '-wd',
        help='Weight decay value used in optimization.',
        default=None if retrain else 1e-4, type=float)
    train_optim_group.add_argument(
        '--dropout',
        help='rate of dropout that will be applied to dropout layers in CNN.',
        default=None if retrain else 0, type=float)
    train_optim_group.add_argument(
        '--patience',
        help='Number of epochs for early stopping patience.',
        type=int, default=None if retrain else 0)
    train_optim_group.add_argument(
        '--tolerance',
        help='Value for the early stopping tolerance.',
        type=float, default=None if retrain else 0.0)
    train_optim_group.add_argument(
        '--accumulation_steps', '-asteps',
        help='Accumulates gradients during the given number of iterations before performing the weight update '
             'in order to virtually increase the size of the batch.',
        default=None if retrain else 1, type=int)
    # train_optim_group.add_argument(
    #     "--loss",
    #     help="Replaces default losses: cross-entropy for CNN and MSE for autoencoders.",
    #     type=str, default=None if retrain else "default",
    #     choices=["default", "L1", "L1Norm", "SmoothL1", "SmoothL1Norm"])

    return train_parent_parser
