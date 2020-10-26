# coding: utf8

import argparse

from clinicadl.tools.deep_learning.iotools import Parameters
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
    # Patch arguments
    'PATCH': '%sPatch-level parameters%s' % (Fore.BLUE, Fore.RESET),
    'PATCH CNN': '%sPatch-level CNN parameters%s' % (Fore.BLUE, Fore.RESET),
    # ROI-based arguments
    'ROI': '%sROI-based parameters%s' % (Fore.BLUE, Fore.RESET),
    'ROI CNN': '%sROI-based CNN parameters%s' % (Fore.BLUE, Fore.RESET),
    # Other optional arguments
    'OPTIONAL': '%sOther options%s' % (Fore.BLUE, Fore.RESET),
}


def set_default_dropout(args):
    if args.dropout is None:
        if args.mode == 'image':
            args.dropout = 0.5
        elif args.mode == 'slice':
            args.dropout = 0.8
        else:
            args.dropout = 0


def extract_tensors(args):
    import sys
    from clinica.utils.stream import FilterOut
    from clinica.pipelines.deeplearning_prepare_data.deeplearning_prepare_data_cli import DeepLearningPrepareDataCLI

    sys.stdout = FilterOut(sys.stdout)

    dl_prepare_data_cli = DeepLearningPrepareDataCLI()
    dl_prepare_data_cli.run_command(args)


def qc_func(args):
    from .quality_check.quality_check import quality_check

    quality_check(
        args.caps_dir,
        args.output_path,
        preprocessing=args.preprocessing,
        tsv_path=args.subjects_sessions_tsv,
        threshold=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.nproc,
        gpu=not args.use_cpu
    )


def generate_data_func(args):
    from .tools.data.generate_data import generate_random_dataset, generate_trivial_dataset

    if args.mode == "random":
        generate_random_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.subjects_sessions_tsv,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            mean=args.mean,
            sigma=args.sigma,
            preprocessing=args.preprocessing)
    else:
        generate_trivial_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.subjects_sessions_tsv,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            preprocessing=args.preprocessing,
            mask_path=args.mask_path,
            atrophy_percent=args.atrophy_percent,
        )


# Function to dispatch training to corresponding function
def train_func(args):
    from .train import train_autoencoder, train_multi_cnn, train_single_cnn

    set_default_dropout(args)

    if args.mode == 'image':
        if args.mode_task == "autoencoder":
            train_params_autoencoder = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_autoencoder.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                visualization=args.visualization,
                verbosity=args.verbose
            )
            train_autoencoder(train_params_autoencoder)
        else:
            train_params_cnn = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_cnn.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                transfer_learning_path=args.transfer_learning_path,
                transfer_learning_selection=args.transfer_learning_selection,
                verbosity=args.verbose
            )
            train_single_cnn(train_params_cnn)
    elif args.mode == 'slice':
        train_params_slice = Parameters(
            args.mode,
            args.tsv_path,
            args.output_dir,
            args.caps_dir,
            args.preprocessing,
            args.model
        )
        train_params_slice.write(
            mri_plane=args.slice_direction,
            diagnoses=args.diagnoses,
            baseline=args.baseline,
            minmaxnormalization=not args.unnormalize,
            learning_rate=args.learning_rate,
            patience=args.patience,
            tolerance=args.tolerance,
            n_splits=args.n_splits,
            split=args.split,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
            evaluation_steps=args.evaluation_steps,
            optimizer='Adam',
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            gpu=not args.use_cpu,
            num_workers=args.nproc,
            selection_threshold=args.selection_threshold,
            prepare_dl=args.use_extracted_slices,
            discarded_slices=args.discarded_slices,
            verbosity=args.verbose
        )
        train_single_cnn(train_params_slice)
    elif args.mode == 'patch':
        if args.mode_task == "autoencoder":
            train_params_autoencoder = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_autoencoder.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                patch_size=args.patch_size,
                stride_size=args.stride_size,
                hippocampus_roi=False,
                visualization=args.visualization,
                prepare_dl=args.use_extracted_patches,
                verbosity=args.verbose
            )
            train_autoencoder(train_params_autoencoder)
        elif args.mode_task == "cnn":
            train_params_patch = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_patch.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                transfer_learning_path=args.transfer_learning_path,
                transfer_learning_selection=args.transfer_learning_selection,
                patch_size=args.patch_size,
                stride_size=args.stride_size,
                hippocampus_roi=False,
                selection_threshold=args.selection_threshold,
                prepare_dl=args.use_extracted_patches,
                verbosity=args.verbose
            )
            train_single_cnn(train_params_patch)
        else:
            train_params_patch = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_patch.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                transfer_learning_path=args.transfer_learning_path,
                transfer_learning_selection=args.transfer_learning_selection,
                patch_size=args.patch_size,
                stride_size=args.stride_size,
                hippocampus_roi=False,
                selection_threshold=args.selection_threshold,
                prepare_dl=args.use_extracted_patches,
                verbosity=args.verbose
            )
            train_multi_cnn(train_params_patch)
    elif args.mode == 'roi':
        if args.mode_task == "autoencoder":
            train_params_autoencoder = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_autoencoder.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                hippocampus_roi=True,
                visualization=args.visualization,
                verbosity=args.verbose
            )
            train_autoencoder(train_params_autoencoder)
        else:
            train_params_patch = Parameters(
                args.mode,
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.preprocessing,
                args.model
            )
            train_params_patch.write(
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                minmaxnormalization=not args.unnormalize,
                n_splits=args.n_splits,
                split=args.split,
                accumulation_steps=args.accumulation_steps,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                gpu=not args.use_cpu,
                batch_size=args.batch_size,
                evaluation_steps=args.evaluation_steps,
                num_workers=args.nproc,
                transfer_learning_path=args.transfer_learning_path,
                transfer_learning_selection=args.transfer_learning_selection,
                hippocampus_roi=True,
                selection_threshold=args.selection_threshold,
                verbosity=args.verbose,
            )
            train_single_cnn(train_params_patch)

    elif args.mode == 'svm':
        raise NotImplementedError("The SVM commandline was not implement yet.")
    else:
        print('Mode not detected in clinicadl')


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
        verbosity=args.verbose
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
        verbosity=args.verbose
    )


def tsv_split_func(args):
    from .tools.tsv.data_split import split_diagnoses

    split_diagnoses(
        args.merged_tsv,
        args.formatted_data_path,
        n_test=args.n_test,
        subset_name=args.subset_name,
        age_name=args.age_name,
        MCI_sub_categories=args.MCI_sub_categories,
        t_val_threshold=args.t_val_threshold,
        p_val_threshold=args.p_val_threshold)


def tsv_kfold_func(args):
    from .tools.tsv.kfold_split import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_splits=args.n_splits,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        verbosity=args.verbose
    )


def tsv_analysis_func(args):
    from .tools.tsv.demographics_analysis import demographics_analysis

    demographics_analysis(
        args.merged_tsv,
        args.formatted_data_path,
        args.results_path,
        diagnoses=args.diagnoses,
        mmse_name=args.mmse_name,
        age_name=args.age_name
    )


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog='clinicadl',
        description='Deep learning software for neuroimaging datasets')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--verbose', '-v', action='count', default=0)

    subparser = parser.add_subparsers(
        title='''Task to execute with clinicadl:''',
        description='''What kind of task do you want to use with clinicadl?
            (tsvtool, preprocessing, generate, train, classify).''',
        dest='task',
        help='''****** Tasks proposed by clinicadl ******''')

    subparser.required = True

    # Generate synthetic data
    generate_parser = subparser.add_parser(
        'generate',
        help='Generate synthetic data for functional tests.'
    )
    generate_parser.add_argument(
        'mode',
        help='Choose which dataset is generated (random, trivial).',
        choices=['random', 'trivial'],
        default='random'
    )
    generate_parser.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None
    )
    generate_parser.add_argument(
        'output_dir',
        help='Folder containing the synthetic dataset.',
        default=None
    )
    generate_parser.add_argument(
        "--subjects_sessions_tsv", "-tsv",
        help='TSV file containing a list of subjects with their sessions.',
        type=str, default=None
    )
    generate_parser.add_argument(
        '--n_subjects',
        type=int,
        default=300,
        help="Number of subjects in each class of the synthetic dataset."
    )
    generate_parser.add_argument(
        '--preprocessing',
        type=str,
        default='t1-linear',
        choices=['t1-linear', 't1-extensive'],
        help="Preprocessing used to generate synthetic data."
    )
    generate_parser.add_argument(
        '--mean',
        type=float,
        default=0,
        help="Mean value of the noise added for the random dataset."
    )
    generate_parser.add_argument(
        '--sigma',
        type=float,
        default=0.5,
        help="Standard deviation of the noise added for the random dataset."
    )
    generate_parser.add_argument(
        '--mask_path',
        type=str,
        help='path to the extracted masks to generate the two labels.',
        default=None
    )
    generate_parser.add_argument(
        '--atrophy_percent',
        type=float,
        default=60,
        help='percentage of atrophy applied'
    )

    generate_parser.set_defaults(func=generate_data_func)

    # Preprocessing
    from clinica.pipelines.t1_linear.t1_linear_cli import T1LinearCLI
    from clinica.engine.cmdparser import init_cmdparser_objects
    from clinicadl.preprocessing.t1_extensive.t1_extensive_cli import T1ExtensiveCli
    preprocessing_parser = subparser.add_parser(
        'preprocessing',
        help='Preprocess T1w-weighted images with t1-linear or t1-extensive pipelines'
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
        'preprocessing',
        help='Preprocessing pipeline on which extraction is performed.',
        choices=['t1-linear']
    )
    clinica_comp.add_argument(
        "caps_directory",
        help='Path to the CAPS directory.'
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
    qc_parser.add_argument("preprocessing",
                           help="Pipeline on which quality check procedure is performed.",
                           type=str,
                           choices=["t1-linear"])
    qc_parser.add_argument("caps_dir",
                           help='Data using CAPS structure.',
                           type=str)
    qc_parser.add_argument("output_path",
                           help="Path to the output tsv file (filename included).",
                           type=str)
    qc_parser.add_argument("--subjects_sessions_tsv", "-tsv",
                           help='TSV file containing a list of subjects with their sessions.',
                           type=str, default=None)
    qc_parser.add_argument("--threshold",
                           help='The threshold on the output probability to decide if the image passed or failed. '
                                '(default=0.5)',
                           type=float, default=0.5)
    qc_parser.add_argument('--batch_size',
                           help='Batch size used in DataLoader (default=1).',
                           default=1, type=int)
    qc_parser.add_argument("-np", "--nproc",
                           help='Number of cores used the quality check. (default=2)',
                           type=int, default=2)
    qc_parser.add_argument('-cpu', '--use_cpu', action='store_true',
                           help='If provided, will use CPU instead of GPU.',
                           default=False)

    qc_parser.set_defaults(func=qc_func)

    # Train - Train CNN model with preprocessed  data
    # train_parser: get command line arguments and options

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

    # Positional arguments
    train_parent_parser = argparse.ArgumentParser(add_help=False)
    train_pos_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"])
    train_pos_group.add_argument(
        'caps_dir',
        help='Data using CAPS structure.',
        default=None)
    train_pos_group.add_argument(
        'preprocessing',
        help='Defines the type of preprocessing of CAPS data.',
        choices=['t1-linear', 't1-extensive'], type=str)
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

    # Computational resources
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

    # Data management
    train_data_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["DATA"])
    train_data_group.add_argument(
        '--diagnoses', '-d',
        help='List of diagnoses that will be selected for training.',
        default=['AD', 'CN'], nargs='+', type=str, choices=['AD', 'CN', 'MCI', 'sMCI', 'pMCI'])
    train_data_group.add_argument(
        '--baseline',
        help='If provided, only the baseline sessions are used for training.',
        action="store_true",
        default=False)
    train_data_group.add_argument(
        '--unnormalize', '-un',
        help='Disable default MinMaxNormalization.',
        action="store_true",
        default=False)

    # Cross-validation
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

    # Optimization parameters
    train_optim_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["OPTIMIZATION"])
    train_optim_group.add_argument(
        '--epochs',
        help='Maximum number of epochs. (default=20)',
        default=20, type=int)
    train_optim_group.add_argument(
        '--learning_rate', '-lr',
        help='Learning rate of the optimization. (default=1e-4)',
        default=1e-4, type=float)
    train_optim_group.add_argument(
        '--weight_decay', '-wd',
        help='Weight decay value used in optimization. (default=1e-4)',
        default=1e-4, type=float)
    train_optim_group.add_argument(
        '--dropout',
        help='rate of dropout that will be applied to dropout layers in CNN. (default=None)',
        default=None, type=float)
    train_optim_group.add_argument(
        '--patience',
        help='Number of epochs for early stopping patience. (default=10)',
        type=int, default=10)
    train_optim_group.add_argument(
        '--tolerance',
        help='Value for the early stopping tolerance. (default=0.0)',
        type=float, default=0.0)
    train_optim_group.add_argument(
        '--evaluation_steps', '-esteps',
        default=0, type=int,
        help='Fix the number of iterations to perform before computing an evaluations. Default will only '
             'perform one evaluation at the end of each epoch. (default=0)')
    train_optim_group.add_argument(
        '--accumulation_steps', '-asteps',
        help='Accumulates gradients during the given number of iterations before performing the weight update '
             'in order to virtually increase the size of the batch. (default=1)',
        default=1, type=int)

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
        dest='mode_task',
        help='''****** Choose a type of network ******''')

    train_image_ae_parser = train_image_subparser.add_parser(
        "autoencoder",
        parents=[
            parent_parser,
            train_parent_parser,
            autoencoder_parent],
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
        help="If transfer_learning from CNN, chooses which best transfer model is selected. "
             "(default=best_balanced_accuracy)",
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
        help='Patch size (default=50)',
        type=int, default=50)
    train_patch_group.add_argument(
        '-ss', '--stride_size',
        help='Stride size (default=50)',
        type=int, default=50)
    train_patch_group.add_argument(
        '--use_extracted_patches',
        help='''If provided the outputs of extract preprocessing are used, else the whole
             MRI is loaded.''',
        default=False, action="store_true")

    train_patch_subparser = train_patch_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or (multi)cnn classification ?''',
        dest='mode_task',
        help='''****** Choose a type of network ******''')
    train_patch_subparser.required = True

    train_patch_ae_parser = train_patch_subparser.add_parser(
        "autoencoder",
        parents=[parent_parser, train_parent_parser, train_patch_parent, autoencoder_parent],
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
        help="If transfer_learning from CNN, chooses which best transfer model is selected. "
             "(default=best_balanced_accuracy)",
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

    train_roi_subparser = train_roi_parser.add_subparsers(
        title='''Task to be performed''',
        description='''Autoencoder reconstruction or cnn classification ?''',
        dest='mode_task',
        help='''****** Choose a type of network ******''')
    train_roi_subparser.required = True

    train_roi_ae_parser = train_roi_subparser.add_parser(
        "autoencoder",
        parents=[
            parent_parser,
            train_parent_parser,
            autoencoder_parent],
        help="Train a ROI-based autoencoder.")

    train_roi_ae_parser.set_defaults(func=train_func)

    train_roi_cnn_parser = train_roi_subparser.add_parser(
        "cnn",
        parents=[
            parent_parser,
            train_parent_parser,
            transfer_learning_parent],
        help="Train a ROI-based CNN.")
    # /!\ If parents list is changed the arguments won't be in the right group anymore !
    train_roi_cnn_parser._action_groups[-1].add_argument(
        '--transfer_learning_selection',
        help="If transfer_learning from CNN, chooses which best transfer model is selected. "
             "(default=best_balanced_accuracy)",
        type=str, default="best_balanced_accuracy", choices=["best_loss", "best_balanced_accuracy"])

    train_roi_cnn_group = train_roi_cnn_parser.add_argument_group(
        TRAIN_CATEGORIES["ROI CNN"])
    train_roi_cnn_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
             subject-level performance. ROIs are selected if their balanced
             accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_roi_cnn_group.set_defaults(func=train_func)

    #########################
    # SLICE
    #########################
    train_slice_parser = train_subparser.add_parser(
        "slice",
        parents=[parent_parser, train_parent_parser],
        help="Train a 2D slice-level CNN.")

    train_slice_group = train_slice_parser.add_argument_group(
        TRAIN_CATEGORIES["SLICE"])
    train_slice_group.add_argument(
        '--slice_direction', '-sd',
        help='''Which coordinate axis to take for slicing the MRI.
                 0 for sagittal
                 1 for coronal
                 2 for axial direction.
                 (default=0)''',
        default=0, type=int)
    train_slice_group.add_argument(
        '--discarded_slices',
        help='''Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides. (default=20)''',
        default=20, type=int, nargs='+'
    )
    train_slice_group.add_argument(
        '--use_extracted_slices',
        help='''If provided the outputs of extract preprocessing are used, else the whole
                 MRI is loaded.''',
        default=False, action="store_true")
    train_slice_group.add_argument(
        '--selection_threshold',
        help='''Threshold on the balanced accuracies to compute the
                 subject-level performance. Slices are selected if their balanced
                 accuracy > threshold. Default corresponds to no selection.''',
        type=float, default=0.0)

    train_slice_parser.set_defaults(func=train_func)

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

    classify_parser.set_defaults(func=classify_func)

    tsv_parser = subparser.add_parser(
        'tsvtool',
        help='''Handle tsv files for metadata processing and data splits''')

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
        nargs="+", type=str, choices=['AD', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])
    tsv_getlabels_subparser.add_argument(
        "--time_horizon",
        help="Time horizon to analyse stability of MCI subjects.",
        default=36, type=int)
    tsv_getlabels_subparser.add_argument(
        "--restriction_path",
        help="Path to a tsv containing the sessions that can be included.",
        type=str, default=None)

    tsv_getlabels_subparser.set_defaults(func=tsv_getlabels_func)

    tsv_split_subparser = tsv_subparser.add_parser(
        'split',
        parents=[parent_parser],
        help='Performs one stratified shuffle split on participant level.')

    tsv_split_subparser.add_argument(
        "merged_tsv",
        help="Path to the file obtained by the command clinica iotools merge-tsv.",
        type=str)
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
        "--age_name",
        help="Name of the variable related to the age in the merged_tsv file.",
        type=str, default="age_bl")
    tsv_split_subparser.add_argument(
        "--MCI_sub_categories",
        help="Deactivate default managing of MCI sub-categories to avoid data leakage.",
        action="store_false", default=True)
    tsv_split_subparser.add_argument(
        "--t_val_threshold", "-t",
        help="The threshold used for the chi2 test on sex distributions.",
        default=0.0642, type=float)
    tsv_split_subparser.add_argument(
        "--p_val_threshold", "-p",
        help="The threshold used for the T-test on age distributions.",
        default=0.80, type=float)
    tsv_split_subparser.add_argument(
        "--subset_name",
        help="Name of the subset that is complementary to train.",
        type=str, default="test")

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
        default=['AD', 'CN'], nargs="+", type=str, choices=['AD', 'CN', 'MCI', 'sMCI', 'pMCI'])
    tsv_analysis_subparser.add_argument(
        "--mmse_name",
        help="Name of the variable related to the MMSE score in the merged_tsv file.",
        type=str, default="MMS")
    tsv_analysis_subparser.add_argument(
        "--age_name",
        help="Name of the variable related to the age in the merged_tsv file.",
        type=str, default="age_bl")

    tsv_analysis_subparser.set_defaults(func=tsv_analysis_func)

    return parser
