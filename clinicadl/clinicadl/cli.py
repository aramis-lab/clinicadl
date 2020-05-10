# coding: utf8

import argparse

from clinicadl.tools.deep_learning.iotools import Parameters


def preprocessing_t1w_func(args):
    from .preprocessing.T1_linear import preprocessing_t1w
    wf = preprocessing_t1w(
            args.bids_dir,
            args.caps_dir,
            args.tsv_file,
            args.working_dir
            )
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})


def extract_data_func(args):
    from .preprocessing.T1_preparedl import extract_dl_t1w
    # from .preprocessing.T1_postprocessing import postprocessing_t1w
    wf = extract_dl_t1w(
            args.caps_dir,
            args.tsv_file,
            args.working_dir,
            args.extract_method,
            args.patch_size,
            args.stride_size,
            args.slice_direction,
            args.slice_mode
            )
    wf.run(plugin='MultiProc', plugin_args={'n_procs': args.nproc})


def generate_data_func(args):
    from .tools.data.generate_data import generate_random_dataset, generate_trivial_dataset

    if args.mode == "random":
        generate_random_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.tsv_path,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            mean=args.mean,
            sigma=args.sigma,
            preprocessing=args.preprocessing,
            output_size=args.output_size)
    else:
        generate_trivial_dataset(
            caps_dir=args.caps_dir,
            tsv_path=args.tsv_path,
            output_dir=args.output_dir,
            n_subjects=args.n_subjects,
            preprocessing=args.preprocessing,
            output_size=args.output_size,
            mask_path=args.mask_path,
            atrophy_percent=args.atrophy_percent,
            group=args.group
        )


# Function to dispatch training to corresponding function
def train_func(args):
    from .subject_level.train_autoencoder import train_autoencoder
    from .subject_level.train_CNN import train_cnn
    from .slice_level.train_CNN import train_slice
    from .patch_level.train_autoencoder import train_autoencoder_patch
    from .patch_level.train_singleCNN import train_patch_single_cnn
    from .patch_level.train_multiCNN import train_patch_multi_cnn

    if args.mode == 'subject':
        if args.train_autoencoder:
            train_params_autoencoder = Parameters(
                    args.tsv_path,
                    args.output_dir,
                    args.caps_dir,
                    args.network
                    )
            train_params_autoencoder.write(
                    args.pretrained_path,
                    args.pretrained_difference,
                    preprocessing=args.preprocessing,
                    diagnoses=args.diagnoses,
                    baseline=args.baseline,
                    minmaxnormalization=args.minmaxnormalization,
                    sampler='random',
                    n_splits=args.n_splits,
                    split=args.split,
                    accumulation_steps=args.accumulation_steps,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    tolerance=args.tolerance,
                    add_sigmoid=args.add_sigmoid,
                    optimizer='Adam',
                    weight_decay=args.weight_decay,
                    gpu=args.use_gpu,
                    batch_size=args.batch_size,
                    evaluation_steps=args.evaluation_steps,
                    num_workers=args.nproc
                    )
            train_autoencoder(train_params_autoencoder)
        else:
            train_params_cnn = Parameters(
                    args.tsv_path,
                    args.output_dir,
                    args.caps_dir,
                    args.network
                    )
            train_params_cnn.write(
                    args.pretrained_path,
                    args.pretrained_difference,
                    preprocessing=args.preprocessing,
                    diagnoses=args.diagnoses,
                    baseline=args.baseline,
                    minmaxnormalization=args.minmaxnormalization,
                    sampler=args.sampler,
                    n_splits=args.n_splits,
                    split=args.split,
                    accumulation_steps=args.accumulation_steps,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    tolerance=args.tolerance,
                    add_sigmoid=args.add_sigmoid,
                    optimizer='Adam',
                    weight_decay=args.weight_decay,
                    gpu=args.use_gpu,
                    batch_size=args.batch_size,
                    evaluation_steps=args.evaluation_steps,
                    num_workers=args.nproc,
                    transfer_learning_path=args.transfer_learning_path,
                    transfer_learning_autoencoder=args.transfer_learning_autoencoder,
                    selection=args.selection
                    )
            train_cnn(train_params_cnn)
    elif args.mode == 'slice':
        train_params_slice = Parameters(
                args.tsv_path,
                args.output_dir,
                args.caps_dir,
                args.network
                )
        train_params_slice.write(
                mri_plane=args.mri_plane,
                diagnoses=args.diagnoses,
                baseline=args.baseline,
                learning_rate=args.learning_rate,
                patience=args.patience,
                tolerance=args.tolerance,
                n_splits=args.n_splits,
                split=args.split,
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimizer='Adam',
                weight_decay=args.weight_decay,
                gpu=args.use_gpu,
                num_workers=args.nproc,
                selection_threshold=args.selection_threshold,
                prepare_dl=args.use_extracted_patches
                )
        train_slice(train_params_slice)
    elif args.mode == 'patch':
        if args.train_autoencoder:
            train_params_autoencoder = Parameters(
                    args.tsv_path,
                    args.output_dir,
                    args.caps_dir,
                    args.network
                    )
            train_params_autoencoder.write(
                    args.pretrained_path,
                    args.pretrained_difference,
                    preprocessing=args.preprocessing,
                    diagnoses=args.diagnoses,
                    baseline=args.baseline,
                    minmaxnormalization=args.minmaxnormalization,
                    sampler='random',
                    n_splits=args.n_splits,
                    split=args.split,
                    accumulation_steps=args.accumulation_steps,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    tolerance=args.tolerance,
                    add_sigmoid=args.add_sigmoid,
                    optimizer='Adam',
                    weight_decay=args.weight_decay,
                    gpu=args.use_gpu,
                    batch_size=args.batch_size,
                    evaluation_steps=args.evaluation_steps,
                    num_workers=args.nproc,
                    patch_size=args.patch_size,
                    patch_stride=args.patch_stride,
                    hippocampus_roi=args.hippocampus_roi,
                    visualization=args.visualization,
                    prepare_dl=args.use_extracted_patches
                    )
            train_autoencoder_patch(train_params_autoencoder)
        else:
            train_params_patch = Parameters(
                    args.tsv_path,
                    args.output_dir,
                    args.caps_dir,
                    args.network
                    )
            train_params_patch.write(
                    args.pretrained_path,
                    args.pretrained_difference,
                    preprocessing=args.preprocessing,
                    diagnoses=args.diagnoses,
                    baseline=args.baseline,
                    minmaxnormalization=args.minmaxnormalization,
                    sampler='random',
                    n_splits=args.n_splits,
                    split=args.split,
                    accumulation_steps=args.accumulation_steps,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    patience=args.patience,
                    tolerance=args.tolerance,
                    add_sigmoid=args.add_sigmoid,
                    optimizer='Adam',
                    weight_decay=args.weight_decay,
                    gpu=args.use_gpu,
                    batch_size=args.batch_size,
                    evaluation_steps=args.evaluation_steps,
                    num_workers=args.nproc,
                    transfer_learning_path=args.transfer_learning_path,
                    transfer_learning_autoencoder=args.transfer_learning_autoencoder,
                    transfer_learning_multicnn=args.transfer_learning_multicnn,
                    selection=args.selection,
                    patch_size=args.patch_size,
                    patch_stride=args.patch_stride,
                    hippocampus_roi=args.hippocampus_roi,
                    selection_threshold=args.selection_threshold,
                    num_cnn=args.num_cnn,
                    prepare_dl=args.use_extracted_patches
                    )
            if args.network_type == 'single':
                train_patch_single_cnn(train_params_patch)
            else:
                train_patch_multi_cnn(train_params_patch)
    elif args.mode == 'svn':
        pass

    else:
        print('Mode not detected in clinicadl')

# Function to dispatch command line options from classify to corresponding
# function


def classify_func(args):
    from .classify.inference import inference_from_model
    # from .preprocessing.T1_postprocessing import postprocessing_t1w
    inference_from_model(
            args.caps_dir,
            args.tsv_file,
            args.output_dir,
            args.model_name,
            )

def parse_command_line():
    parser = argparse.ArgumentParser(
            prog='clinicadl',
            description='Clinica Deep Learning.')

    parser.add_argument('--verbose', '-v', action='count')

    subparser = parser.add_subparsers(
            title='''Task to execute with clinicadl:''',
            description='''What kind of task do you want to use with clinicadl?
            (preprocessing, extract, generate, train, validate, classify).''',
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
            'tsv_path',
            help='TSV path with subjects/sessions to use for data generation.',
            default=None
            )
    generate_parser.add_argument(
            'output_dir',
            help='Folder containing the synthetic dataset.',
            default=None
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
            default='linear',
            help="Preprocessing used to generate synthetic data."
            )
    generate_parser.add_argument(
            '--output_size',
            type=int,
            nargs="+",
            default=None,
            help="If a value is given, interpolation will be used to up/downsample the image."
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
    generate_parser.add_argument(
            '--group',
            type=str,
            default=None,
            help="Specific argument for dartel preprocessing."
            )

    generate_parser.set_defaults(func=generate_data_func)

    # Preprocessing 1
    # preprocessing_parser: get command line arguments and options for
    # preprocessing

    preprocessing_parser = subparser.add_parser(
            'preprocessing',
            help='Prepare data for training (needs clinica installed).')
    preprocessing_parser.add_argument(
            'bids_dir',
            help='Data using BIDS structure.',
            default=None
            )
    preprocessing_parser.add_argument(
            'caps_dir',
            help='Data using CAPS structure.',
            default=None
            )
    preprocessing_parser.add_argument(
            'tsv_file',
            help='TSV file with subjects/sessions to process.',
            default=None
            )
    preprocessing_parser.add_argument(
            'working_dir',
            help='Working directory to save temporary file.',
            default=None
            )
    preprocessing_parser.add_argument(
            '-np', '--nproc',
            help='Number of cores used for processing (2 by default)',
            type=int, default=2
            )

    preprocessing_parser.set_defaults(func=preprocessing_t1w_func)

    # Preprocessing 2 - Extract data: slices or patches
    # extract_parser: get command line argument and options

    extract_parser = subparser.add_parser(
            'extract',
            help='Create data (slices or patches) for training.'
            )
    extract_parser.add_argument(
            'caps_dir',
            help='Data using CAPS structure.',
            default=None
            )
    extract_parser.add_argument(
            'tsv_file',
            help='TSV file with subjects/sessions to process.',
            default=None
            )
    extract_parser.add_argument(
            'working_dir',
            help='Working directory to save temporary file.',
            default=None
            )
    extract_parser.add_argument(
            'extract_method',
            help='''Method used to extract features. Three options:
                 'slice' to get 2D slices from the MRI,
                 'patch' to get 3D volumetric patches or
                 'whole' to get the complete MRI.''',
            choices=['slice', 'patch', 'whole'], default='whole'
            )
    extract_parser.add_argument(
            '-psz', '--patch_size',
            help='''Patch size (only for 'patch' extraction) e.g: --patch_size 50''',
            type=int, default=50
            )
    extract_parser.add_argument(
            '-ssz', '--stride_size',
            help='''Stride size (only for 'patch' extraction) e.g.: --stride_size 50''',
            type=int, default=50
            )
    extract_parser.add_argument(
            '-sd', '--slice_direction',
            help='''Slice direction (only for 'slice' extraction). Three options:
                 '0' -> Sagittal plane,
                 '1' -> Coronal plane or
                 '2' -> Axial plane''',
            type=int, default=0
            )
    extract_parser.add_argument(
            '-sm', '--slice_mode',
            help='''Slice mode (only for 'slice' extraction). Two options:
                 'original' to save one single channel (intensity),
                 'rgb' to saves three channel (with same intensity).''',
            choices=['original', 'rgb'], default='rgb'
            )
    extract_parser.add_argument(
            '-np', '--nproc',
            help='Number of cores used for processing',
            type=int, default=2
            )

    extract_parser.set_defaults(func=extract_data_func)

    # Train - Train CNN model with preprocessed  data
    # train_parser: get command line arguments and options

    train_parser = subparser.add_parser(
            'train',
            help='Train with your data and create a model.')
    train_parser.add_argument(
            'mode',
            help='Choose your mode (subject level, slice level, patch level, svm).',
            choices=['subject', 'slice', 'patch', 'svm'],
            default='subject')
    train_parser.add_argument(
            'caps_dir',
            help='Data using CAPS structure.',
            default=None)
    train_parser.add_argument(
            'tsv_path',
            help='TSV path with subjects/sessions to process.',
            default=None)
    train_parser.add_argument(
            'output_dir',
            help='Folder containing results of the training.',
            default=None)
    train_parser.add_argument(
            'network',
            help='CNN Model to be used during the training.',
            default='Conv5_FC3')

    # Optional parameters
    # Computational issues
    train_parser.add_argument(
            '-gpu', '--use_gpu', action='store_true',
            help='Uses GPU instead of CPU if CUDA is available',
            default=False)
    train_parser.add_argument(
            '-np', '--nproc',
            help='Number of cores used during the training.',
            type=int, default=2)
    train_parser.add_argument(
            '--visualization',
            help='Save results in visualization folder',
            action="store_true",
            default=False)
    train_parser.add_argument(
            '--batch_size',
            default=2, type=int,
            help='Batch size for training. (default=2)',)
    train_parser.add_argument(
            '--evaluation_steps', '-esteps',
            default=1, type=int,
            help='Fix the number of batches to use before validation.')

    # Data Management
    train_parser.add_argument(
            '--preprocessing',
            help='Defines the type of preprocessing of CAPS data.',
            choices=['linear', 'mni'], type=str,
            default='linear')
    train_parser.add_argument(
            '--diagnoses', '-d',
            help='Take all the subjects possible for autoencoder training.',
            default=['AD', 'CN'], nargs='+', type=str)
    train_parser.add_argument(
            '--baseline',
            help='if True only the baseline is used.',
            action="store_true",
            default=False)
    train_parser.add_argument(
            '--minmaxnormalization', '-n',
            help='Performs MinMaxNormalization.',
            action="store_true",
            default=False)

    # Cross-validation
    train_parser.add_argument(
            '--n_splits',
            help='If a value is given will load data of a k-fold CV.',
            type=int, default=None)
    train_parser.add_argument(
            '--split',
            help='Will load the specific split wanted.',
            type=int, default=0)

    # Training arguments
    train_parser.add_argument(
            '--network_type',
            help='Chose between sinlge or multi CNN (applies only for mode patch-level)',
            choices=['single', 'multi'], type=str,
            default='single')
    train_parser.add_argument(
            '-tAE', '--train_autoencoder',
            help='Add this option if you want to train an autoencoder.',
            action="store_true",
            default=False)
    train_parser.add_argument(
            '-hroi', '--hippocampus_roi',
            help='If true, use the hippocampus region.',
            action="store_true",
            default=False)
    train_parser.add_argument(
            '--num_cnn',
            help='''How many CNNs we want to train in a patch-wise way.
                 By default, we train each patch from all subjects for one CNN''',
            default=36, type=int)
    train_parser.add_argument(
            '--mri_plane',
            help='''Which coordinate axis to take for slicing the MRI.
                 0 for sagittal
                 1 for coronal
                 2 for axial direction.''',
            default=0, type=int)
    train_parser.add_argument(
            '--sampler', '-sm',
            help='Sampler to be used.',
            default='random', type=str)
    train_parser.add_argument(
            '--accumulation_steps', '-asteps',
            help='Accumulates gradients in order to increase the size of the batch.',
            default=1, type=int)
    train_parser.add_argument(
            '--epochs',
            help='Epochs through the data. (default=20)',
            default=20, type=int)
    train_parser.add_argument(
            '--learning_rate', '-lr',
            help='Learning rate of the optimization. (default=0.01)',
            default=1e-4, type=float)
    train_parser.add_argument(
            '--weight_decay', '-wd',
            help='Weight decay value used in optimization. (default=1e-4)',
            default=1e-4, type=float)
    train_parser.add_argument(
            '--patience',
            help='Waiting time for early stopping.',
            type=int, default=10)
    train_parser.add_argument(
            '--tolerance',
            help='Tolerance value for the early stopping.',
            type=float, default=0.0)
    train_parser.add_argument(
            '--add_sigmoid',
            help='Ad sigmoid function at the end of the decoder.',
            default=False, action="store_true")
    train_parser.add_argument(
            '-psz', '--patch_size',
            help='Patch size e.g: --patch_size 50',
            type=int, default=50)
    train_parser.add_argument(
            '-pst', '--patch_stride',
            help='Patch stride e.g: --patch_stride 50',
            type=int, default=50)

    # Transfer learning from other autoencoder/network
    train_parser.add_argument(
            '--pretrained_path',
            help='Path to a pretrained model (can be of different size).',
            type=str, default=None)
    train_parser.add_argument(
            '--pretrained_difference',
            help='''Difference of size between the pretrained autoencoder and
                 the training one. If the new one is larger, difference will be
                 positive.''',
            type=int, default=0)
    train_parser.add_argument(
            '--transfer_learning_path',
            help="If an existing path is given, a pretrained autoencoder is used.",
            type=str, default=None)
    train_parser.add_argument(
            '--transfer_learning_autoencoder',
            help='''If do transfer learning using an autoencoder else will look
                 for a CNN model.''',
            default=False, action="store_true")
    train_parser.add_argument(
            '--transfer_learning_multicnn',
            help='''If do transfer learning for multi-CNN initialize with
                 multiple CNN models.''',
            default=False, action="store_true")
    train_parser.add_argument(
            '--selection',
            help="Allow to choose which model of the experiment is loaded.",
            type=str, default="best_acc", choices=["best_loss", "best_acc"])
    train_parser.add_argument(
            '--selection_threshold',
            help='''Threshold on the balanced accuracies to compute the
                 subject_level performance.only based on patches with balanced
                 accuracy > threshold.''',
            type=float, default=0.0)
    train_parser.add_argument(
            '--use_extracted_patches',
            help='''If True the outputs of preprocessing are used, else the whole
                 MRI is loaded.''',
            default=False, action="store_true")
    train_parser.set_defaults(func=train_func)

    # Classify - Classify a subject or a list of tesv files with the CNN
    # provieded as argument.
    # classify_parser: get command line arguments and options

    classify_parser = subparser.add_parser(
            'classify',
            help='''Classify one image or a list of images with your previously
                 trained model.''')
    classify_parser.add_argument(
            'caps_dir',
            help='Data using CAPS structure.',
            default=None)
    classify_parser.add_argument(
            'tsv_path',
            help='TSV path with subjects/sessions to process.',
            default=None)
    classify_parser.add_argument(
            'output_dir',
            help='Folder containing results of the training.',
            default=None)
    classify_parser.add_argument(
            'model_name',
            help='Model to use for classification.',
            choices=['2D_slice', '3D_patch_1', '3D_patch_2', 'subject_1', 'subject_2'],
            default='2D_slice')

    classify_parser.set_defaults(func=classify_func)

#    args = parser.parse_args()
#
#    commandline = parser.parse_known_args()
#
#    if args.train_autoencoder:
#      model_type = 'autoencoder'
#    else:
#      model_type = 'cnn'
#
#    commandline_to_json(commandline, model_type)
#
#    print(args)

    return parser
