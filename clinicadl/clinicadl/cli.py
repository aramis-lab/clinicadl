# coding: utf8

import argparse

from colorama import Fore

TRAIN_CATEGORIES = {
    # General parent group
    "POSITIONAL": "%sPositional arguments%s" % (Fore.BLUE, Fore.RESET),
    "MODEL": "%sModel specific arguments%s" % (Fore.BLUE, Fore.RESET),
    "COMPUTATIONAL": "%sComputational resources%s" % (Fore.BLUE, Fore.RESET),
    "DATA": "%sData management%s" % (Fore.BLUE, Fore.RESET),
    "CROSS-VALIDATION": "%sCross-validation arguments%s" % (Fore.BLUE, Fore.RESET),
    "OPTIMIZATION": "%sOptimization parameters%s" % (Fore.BLUE, Fore.RESET),
    # Other parent groups
    "TRANSFER LEARNING": "%sTransfer learning%s" % (Fore.BLUE, Fore.RESET),
    # Tasks
    "RECONSTRUCTION": "%sReconstruction specific%s" % (Fore.BLUE, Fore.RESET),
    "CLASSIFICATION": "%sClassification specific%s" % (Fore.BLUE, Fore.RESET),
    "REGRESSION": "%sRegression specific%s" % (Fore.BLUE, Fore.RESET),
    # Slice-level
    "SLICE": "%sSlice-level parameters%s" % (Fore.BLUE, Fore.RESET),
    # Patch arguments
    "PATCH": "%sPatch-level parameters%s" % (Fore.BLUE, Fore.RESET),
    # ROI-based arguments
    "ROI": "%sROI-based parameters%s" % (Fore.BLUE, Fore.RESET),
    # Other optional arguments
    "OPTIONAL": "%sOther options%s" % (Fore.BLUE, Fore.RESET),
}


def rs_func(args):
    from clinicadl.random_search.random_search import launch_search
    from clinicadl.utils.meta_maps.random_search_analysis import random_search_analysis

    if args.random_task == "generate":
        launch_search(args)
    elif args.random_task == "analysis":
        random_search_analysis(
            args.launch_dir,
        )
    else:
        raise ValueError("This task was not implemented in random-search.")


def retrain_func(args):
    from .train.train_from_json import retrain

    retrain(args.json_path, args.output_dir, verbose=args.verbose)


def resume_func(args):
    from clinicadl.train.resume import automatic_resume

    if args.use_cpu and args.use_gpu:
        raise ValueError(
            "The flags --use_cpu and --use_gpu cannot be specified at the same time."
        )
    elif args.use_cpu:
        gpu = False
    elif args.use_gpu:
        gpu = True
    else:
        gpu = None

    automatic_resume(
        model_path=args.model_path,
        verbose=args.verbose,
    )


# Function to dispatch training to corresponding function
def train_func(args):
    from .train import train

    train(args, erase_existing=False)


# Function to dispatch command line options from predict to corresponding
# function
def classify_func(args):
    from clinicadl.infer.infer import classify

    predict_cli(
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
        multi_cohort=args.multi_cohort,
    )


# Functions to dispatch command line options from tsvtool to corresponding
# function
def tsv_restrict_func(args):
    from clinicadl.tsvtools.restrict import aibl_restriction, oasis_restriction

    if args.dataset == "AIBL":
        aibl_restriction(args.merged_tsv, args.results_path)
    elif args.dataset == "OASIS":
        oasis_restriction(args.merged_tsv, args.results_path)


def tsv_getlabels_func(args):
    from clinicadl.tsvtools.getlabels import get_labels

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
        verbose=args.verbose,
    )


def tsv_split_func(args):
    from clinicadl.tsvtools.split import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_test=args.n_test,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        p_age_threshold=args.p_age_threshold,
        p_sex_threshold=args.p_sex_threshold,
        ignore_demographics=args.ignore_demographics,
        verbose=args.verbose,
        categorical_split_variable=args.categorical_split_variable,
    )


def tsv_kfold_func(args):
    from clinicadl.tsvtools.kfold import split_diagnoses

    split_diagnoses(
        args.formatted_data_path,
        n_splits=args.n_splits,
        subset_name=args.subset_name,
        MCI_sub_categories=args.MCI_sub_categories,
        stratification=args.stratification,
        verbose=args.verbose,
    )


def tsv_analysis_func(args):
    from clinicadl.tsvtools.analysis import demographics_analysis

    demographics_analysis(
        args.merged_tsv,
        args.formatted_data_path,
        args.results_path,
        diagnoses=args.diagnoses,
    )


def interpret_func(args):
    from .interpret.interpret import interpret_cli

    interpret_cli(args)


def parse_command_line():
    parser = argparse.ArgumentParser(
        prog="clinicadl", description="Deep learning software for neuroimaging datasets"
    )

    parser.add_argument(
        "-l",
        "--logname",
        dest="logname",
        default="clinicaDL.log",
        metavar="file.log",
        help="Define the log file name (default: clinicaDL.log)",
    )
    parser.add_argument(
        "-V",
        "--version",
        dest="version",
        action="store_true",
        default=False,
        help="ClinicaDL's installed version",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--verbose", "-v", action="count", default=0)

    subparser = parser.add_subparsers(
        title="""Task to execute with clinicadl""",
        description="""What kind of task do you want to use with clinicadl?""",
        dest="task",
        help="""****** Tasks proposed by clinicadl ******""",
    )

    # random search parsers
    rs_parser = subparser.add_parser(
        "random-search",
        parents=[parent_parser],
        help="Generate random networks to explore hyper parameters space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rs_subparsers = rs_parser.add_subparsers(
        title="""Possibilities for random network training""",
        description="""You can generate and train a new random network,
        or relaunch a previous random job with some alterations.""",
        dest="random_task",
        help="""****** Possible tasks ******""",
    )

    rs_subparsers.required = True

    rs_generate_parser = rs_subparsers.add_parser(
        "generate",
        parents=[parent_parser],
        help="Sample a new network and train it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rs_pos_group = rs_generate_parser.add_argument_group(TRAIN_CATEGORIES["POSITIONAL"])
    rs_pos_group.add_argument(
        "launch_dir", type=str, help="Directory containing the random_search.json file."
    )
    rs_pos_group.add_argument("name", type=str, help="Name of the job.")

    rs_comp_group = rs_generate_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"]
    )
    rs_comp_group.add_argument(
        "-cpu",
        "--use_cpu",
        action="store_true",
        help="If provided, will use CPU instead of GPU.",
        default=False,
    )
    rs_comp_group.add_argument(
        "-np",
        "--nproc",
        help="Number of cores used during the training.",
        type=int,
        default=2,
    )
    rs_comp_group.add_argument(
        "--batch_size", default=2, type=int, help="Batch size for training."
    )
    rs_comp_group.add_argument(
        "--evaluation_steps",
        "-esteps",
        default=0,
        type=int,
        help="Fix the number of iterations to perform before computing an evaluation. Default will only "
        "perform one evaluation at the end of each epoch.",
    )

    rs_generate_parser.set_defaults(func=rs_func)

    rs_analysis_parser = rs_subparsers.add_parser(
        "analysis",
        help="Performs the analysis of all jobs in launch_dir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    rs_analysis_parser.add_argument(
        "launch_dir", type=str, help="Directory containing the random_search.json file."
    )

    rs_analysis_parser.set_defaults(func=rs_func)

    train_parser = subparser.add_parser(
        "train", help="Train with your data and create a model."
    )

    train_subparser = train_parser.add_subparsers(
        title="""Inputs types implemented in clinicadl""",
        description="""What type of input do you want to use?
                (image, patch, roi, slice).""",
        dest="mode",
        help="""****** Input types proposed by clinicadl ******""",
    )

    train_subparser.required = True

    # Transfer learning
    transfer_learning_parent = argparse.ArgumentParser(add_help=False)
    transfer_learning_group = transfer_learning_parent.add_argument_group(
        TRAIN_CATEGORIES["TRANSFER LEARNING"]
    )
    transfer_learning_group.add_argument(
        "--transfer_learning_path",
        help="If an existing path is given, a pretrained model is used.",
        type=str,
        default=None,
    )
    transfer_learning_group.add_argument(
        "--transfer_learning_selection",
        help="If transfer_learning from CNN, chooses which best transfer model is selected.",
        type=str,
        default="loss",
    )

    # Reconstruction
    reconstruction_parent = argparse.ArgumentParser(add_help=False)
    reconstruction_group = reconstruction_parent.add_argument_group(
        TRAIN_CATEGORIES["RECONSTRUCTION"]
    )
    reconstruction_group.add_argument(
        "--visualization",
        help="Save examples of image reconstructions.",
        action="store_true",
        default=False,
    )

    # Classification
    classification_parent = argparse.ArgumentParser(add_help=False)
    classification_group = classification_parent.add_argument_group(
        TRAIN_CATEGORIES["CLASSIFICATION"]
    )
    classification_group.add_argument(
        "--label",
        default="diagnosis",
        help="Target label of the classification task. Must correspond to a categorical variable.",
    )
    classification_group.add_argument(
        "--selection_threshold",
        help="""Threshold on the balanced accuracies to compute the
                image-level performance. Parts are selected if their balanced
                accuracy > threshold. Default corresponds to no selection.""",
        type=float,
        default=0.0,
    )

    # Regression
    regression_parent = argparse.ArgumentParser(add_help=False)
    regression_group = regression_parent.add_argument_group(
        TRAIN_CATEGORIES["REGRESSION"]
    )
    regression_group.add_argument(
        "--label",
        default="age",
        help="Target label of the regression task. Must correspond to a continuous variable.",
    )

    ######################
    # IMAGE
    ######################
    train_image_parser = train_subparser.add_parser(
        "image", help="Train a 3D image-level network."
    )

    train_image_subparser = train_image_parser.add_subparsers(
        title="""Task to be performed""",
        dest="network_task",
        help="""****** Choose a task ******""",
    )

    train_parent_parser = return_train_parent_parser()
    train_image_rec_parser = train_image_subparser.add_parser(
        "reconstruction",
        parents=[
            parent_parser,
            train_parent_parser,
            reconstruction_parent,
            transfer_learning_parent,
        ],
        help="Reconstruction of the input image.",
    )

    train_image_rec_parser.set_defaults(func=train_func)

    train_image_cla_parser = train_image_subparser.add_parser(
        "classification",
        parents=[
            parent_parser,
            train_parent_parser,
            classification_parent,
            transfer_learning_parent,
        ],
        help="Classification of the target label.",
    )

    train_image_cla_parser.set_defaults(func=train_func)

    train_image_reg_parser = train_image_subparser.add_parser(
        "regression",
        parents=[
            parent_parser,
            train_parent_parser,
            regression_parent,
            transfer_learning_parent,
        ],
        help="Regression of the target label.",
    )

    train_image_reg_parser.set_defaults(func=train_func)

    #########################
    # PATCH
    #########################
    train_patch_parser = train_subparser.add_parser(
        "patch", help="Train a 3D patch-level network."
    )

    train_patch_parent = argparse.ArgumentParser(add_help=False)
    train_patch_group = train_patch_parent.add_argument_group(TRAIN_CATEGORIES["PATCH"])
    train_patch_group.add_argument(
        "-ps", "--patch_size", help="Patch size", type=int, default=50
    )
    train_patch_group.add_argument(
        "-ss", "--stride_size", help="Stride size", type=int, default=50
    )
    train_patch_group.add_argument(
        "--use_extracted_patches",
        help="""If provided the outputs of extract preprocessing are used, else the whole
             MRI is loaded.""",
        default=False,
        action="store_true",
    )

    train_patch_subparser = train_patch_parser.add_subparsers(
        title="""Task to be performed""",
        dest="network_task",
        help="""****** Choose a task ******""",
    )
    train_patch_subparser.required = True

    train_patch_ae_parser = train_patch_subparser.add_parser(
        "reconstruction",
        parents=[
            parent_parser,
            train_parent_parser,
            train_patch_parent,
            reconstruction_parent,
            transfer_learning_parent,
        ],
        help="Reconstruction of the input patches.",
    )

    train_patch_ae_parser.set_defaults(func=train_func)

    train_patch_cla_parser = train_patch_subparser.add_parser(
        "classification",
        parents=[
            parent_parser,
            train_parent_parser,
            train_patch_parent,
            classification_parent,
            transfer_learning_parent,
        ],
        help="Classification of the target label.",
    )

    train_patch_cla_parser.set_defaults(func=train_func)

    train_patch_reg_parser = train_patch_subparser.add_parser(
        "regression",
        parents=[
            parent_parser,
            train_parent_parser,
            train_patch_parent,
            regression_parent,
            transfer_learning_parent,
        ],
        help="Regression of the target label.",
    )
    train_patch_reg_parser.set_defaults(func=train_func)

    #########################
    # ROI
    #########################
    train_roi_parser = train_subparser.add_parser(
        "roi", help="Train a ROI-based level network."
    )

    train_roi_parent = argparse.ArgumentParser(add_help=False)
    train_roi_group = train_roi_parent.add_argument_group(TRAIN_CATEGORIES["ROI"])
    train_roi_group.add_argument(
        "-rl",
        "--roi_list",
        help="Names of the regions used for the classification task."
        "Default will use the hippocampi as described in (Wen et al, 2019).",
        type=str,
        nargs="+",
        default=None,
    )
    train_roi_group.add_argument(
        "--uncropped_roi",
        help="If given the image is as large as the whole image. Default will crop the image"
        "with the smallest bounding box possible.",
        action="store_true",
        default=False,
    )
    train_roi_group.add_argument(
        "--use_extracted_roi",
        help="""If provided the outputs of extract preprocessing are used, else the whole
                 MRI is loaded.""",
        default=False,
        action="store_true",
    )

    train_roi_subparser = train_roi_parser.add_subparsers(
        title="""Task to be performed""",
        dest="network_task",
        help="""****** Choose a task ******""",
    )
    train_roi_subparser.required = True

    train_roi_rec_parser = train_roi_subparser.add_parser(
        "reconstruction",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            reconstruction_parent,
            transfer_learning_parent,
        ],
        help="Reconstruction of the input regions.",
    )

    train_roi_rec_parser.set_defaults(func=train_func)

    train_roi_cla_parser = train_roi_subparser.add_parser(
        "classification",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            classification_parent,
            transfer_learning_parent,
        ],
        help="Classification of the target label.",
    )

    train_roi_cla_parser.set_defaults(func=train_func)

    train_roi_reg_parser = train_roi_subparser.add_parser(
        "regression",
        parents=[
            parent_parser,
            train_parent_parser,
            train_roi_parent,
            regression_parent,
            transfer_learning_parent,
        ],
        help="Regression of the target label.",
    )

    train_roi_reg_parser.set_defaults(func=train_func)

    #########################
    # SLICE
    #########################
    train_slice_parser = train_subparser.add_parser(
        "slice", help="Train a 2D slice-level network."
    )

    train_slice_subparser = train_slice_parser.add_subparsers(
        title="""Task to be performed""",
        dest="network_task",
        help="""****** Choose a task ******""",
    )
    train_slice_subparser.required = True

    train_slice_parent = argparse.ArgumentParser(add_help=False)
    train_slice_group = train_slice_parent.add_argument_group(TRAIN_CATEGORIES["SLICE"])
    train_slice_group.add_argument(
        "--slice_direction",
        "-sd",
        help="""Which coordinate axis to take for slicing the MRI.
                 0 for sagittal
                 1 for coronal
                 2 for axial direction.""",
        default=0,
        type=int,
    )
    train_slice_group.add_argument(
        "--discarded_slices",
        help="""Number of slices discarded from respectively the beginning and
        the end of the MRI volume.  If only one argument is given, it will be
        used for both sides.""",
        default=20,
        type=int,
        nargs="+",
    )
    train_slice_group.add_argument(
        "--use_extracted_slices",
        help="""If provided the outputs of extract preprocessing are used, else the whole
                 MRI is loaded.""",
        default=False,
        action="store_true",
    )

    train_slice_rec_parser = train_slice_subparser.add_parser(
        "reconstruction",
        parents=[
            parent_parser,
            train_parent_parser,
            train_slice_parent,
            reconstruction_parent,
            transfer_learning_parent,
        ],
        help="Reconstruction of the input slices.",
    )

    train_slice_rec_parser.set_defaults(func=train_func)

    train_slice_cla_parser = train_slice_subparser.add_parser(
        "classification",
        parents=[
            parent_parser,
            train_parent_parser,
            train_slice_parent,
            classification_parent,
            transfer_learning_parent,
        ],
        help="Classification of the target label.",
    )

    train_slice_cla_parser.set_defaults(func=train_func)

    train_slice_reg_parser = train_slice_subparser.add_parser(
        "regression",
        parents=[
            parent_parser,
            train_parent_parser,
            train_slice_parent,
            regression_parent,
            transfer_learning_parent,
        ],
        help="Regression of the target label.",
    )

    train_slice_reg_parser.set_defaults(func=train_func)

    #########################
    # FROM JSON
    #########################
    train_json_parser = train_subparser.add_parser(
        "from_json",
        parents=[parent_parser],
        help="Train a network as defined in a JSON file.",
    )
    train_json_group = train_json_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"]
    )
    train_json_group.add_argument("json_path", type=str, help="Path to the JSON file.")
    train_json_group.add_argument(
        "output_dir", type=str, help="Directory in which the new job is stored."
    )

    train_json_parser.set_defaults(func=retrain_func)

    #########################
    # RESUME
    #########################
    resume_parser = train_subparser.add_parser(
        "resume",
        parents=[parent_parser],
        help="Resume all jobs prematurely ended in launch_dir.",
    )

    resume_parser.add_argument(
        "model_path", type=str, help="Directory containing the random_search.json file."
    )

    resume_parser.set_defaults(func=resume_func)

    # Predict - Predict the output value of a subject or a list defined in a TSV file
    # provided as argument.

    predict_parser = subparser.add_parser(
        "predict",
        parents=[parent_parser],
        help="""Performs the individual predictions of a list of subject in tsv_path. 
        If labels are given, will also compute global metrics on the data set.""",
    )
    predict_pos_group = predict_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"]
    )
    predict_pos_group.add_argument(
        "caps_directory", help="Data using CAPS structure.", default=None
    )
    predict_pos_group.add_argument(
        "tsv_path",
        help="""Path to the file with subjects/sessions to process.
        If it includes the filename will load the tsv file directly.
        Else will load the baseline tsv files of wanted diagnoses produced by tsvtool.""",
        default=None,
    )
    predict_pos_group.add_argument(
        "model_path",
        help="""Path to the folder where the model is stored. Folder structure
                should be the same obtained during the training.""",
        default=None,
    )
    predict_pos_group.add_argument(
        "prefix_output",
        help="Prefix to name the files resulting from the prediction task.",
        type=str,
    )

    # Computational resources
    predict_comput_group = predict_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"]
    )
    predict_comput_group.add_argument(
        "-cpu",
        "--use_cpu",
        action="store_true",
        help="Uses CPU instead of GPU.",
        default=False,
    )
    predict_comput_group.add_argument(
        "-np",
        "--nproc",
        help="Number of cores used during the task.",
        type=int,
        default=2,
    )
    predict_comput_group.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch size for data loading. (default=2)",
    )

    # Specific classification arguments
    predict_specific_group = predict_parser.add_argument_group(
        TRAIN_CATEGORIES["OPTIONAL"]
    )
    predict_specific_group.add_argument(
        "-nl",
        "--no_labels",
        action="store_true",
        help="Add this flag if your dataset does not contain a ground truth.",
        default=False,
    )
    predict_specific_group.add_argument(
        "--use_extracted_features",
        help="""If True the extract slices or patche are used, otherwise the they
                will be extracted on the fly (if necessary).""",
        default=False,
        action="store_true",
    )
    predict_specific_group.add_argument(
        "--selection_metrics",
        help="""List of metrics to find the best models to evaluate. Default will
        perform a prediction on the best model based on the loss.""",
        default=["loss"],
        nargs="+",
    )
    predict_specific_group.add_argument(
        "--diagnoses",
        help="List of participants that will be classified.",
        nargs="+",
        type=str,
        choices=["AD", "CN", "MCI", "sMCI", "pMCI"],
        default=None,
    )
    predict_specific_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort classification. In this case, caps_directory and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False,
    )

    predict_parser.set_defaults(func=predict_func)

    interpret_parent_parser = argparse.ArgumentParser(add_help=False)

    interpret_pos_group = interpret_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"]
    )
    interpret_pos_group.add_argument(
        "model_path", type=str, help="Path to the model output directory."
    )
    interpret_pos_group.add_argument(
        "name", type=str, help="Name of the interpretation map."
    )

    interpret_comput_group = interpret_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"]
    )
    interpret_comput_group.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size for selection of images (keep_true).",
    )
    interpret_comput_group.add_argument(
        "-cpu",
        "--use_cpu",
        action="store_true",
        default=False,
        help="Uses gpu instead of cpu if cuda is available.",
    )
    interpret_comput_group.add_argument(
        "-np",
        "--nproc",
        default=2,
        type=int,
        help="the number of batches being loaded in parallel.",
    )

    interpret_model_group = interpret_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["MODEL"]
    )
    interpret_model_group.add_argument(
        "--selection_metrics",
        default=["loss"],
        type=str,
        nargs="+",
        help="Loads the model selected on the metrics given.",
    )

    interpret_data_group = interpret_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["DATA"]
    )
    interpret_data_group.add_argument(
        "--tsv_path",
        type=str,
        default=None,
        help="TSV path with subjects/sessions to process, if different from classification task.",
    )
    interpret_data_group.add_argument(
        "--caps_directory",
        type=str,
        default=None,
        help="Data using CAPS structure, if different from classification task",
    )
    interpret_data_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort interpretation. In this case, caps_directory and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False,
    )
    interpret_data_group.add_argument(
        "--diagnosis",
        "-d",
        default="AD",
        type=str,
        help="The images corresponding to this diagnosis only will be loaded.",
    )
    interpret_data_group.add_argument(
        "--target_node",
        default=0,
        type=str,
        help="Which target node the gradients explain. Default takes the first output node.",
    )
    interpret_data_group.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="If provided, only the baseline sessions are used for training.",
    )
    interpret_data_group.add_argument(
        "--save_individual",
        type=str,
        default=None,
        help="Saves individual saliency maps in addition to the mean saliency map.",
    )

    interpret_parser = subparser.add_parser(
        "interpret",
        parents=[parent_parser, interpret_parent_parser],
        help="""Interpret the prediction of a CNN with saliency maps.""",
    )

    interpret_parser.set_defaults(func=interpret_func)

    return parser


def return_train_parent_parser():
    # Main train parent parser common to train and random search
    train_parent_parser = argparse.ArgumentParser(add_help=False)
    train_pos_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["POSITIONAL"]
    )
    train_pos_group.add_argument(
        "caps_directory", help="Data using CAPS structure.", default=None
    )
    train_pos_group.add_argument(
        "preprocessing",
        help="Defines the type of preprocessing of CAPS data.",
        choices=["t1-linear", "t1-extensive", "t1-volume"],
        type=str,
    )
    train_pos_group.add_argument(
        "tsv_path", help="TSV path with subjects/sessions to process.", default=None
    )
    train_pos_group.add_argument(
        "output_dir", help="Folder containing results of the training.", default=None
    )
    train_pos_group.add_argument(
        "model", help="CNN Model to be used during the training.", default="Conv5_FC3"
    )

    train_model_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["MODEL"]
    )
    train_model_group.add_argument(
        "--multi",
        action="store_true",
        help="If provided uses a multi-network framework.",
        default=False,
    )

    train_comput_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["COMPUTATIONAL"]
    )
    train_comput_group.add_argument(
        "-cpu",
        "--use_cpu",
        action="store_true",
        help="If provided, will use CPU instead of GPU.",
        default=False,
    )
    train_comput_group.add_argument(
        "-np",
        "--nproc",
        help="Number of cores used during the training. (default=2)",
        type=int,
        default=2,
    )
    train_comput_group.add_argument(
        "--batch_size", default=2, type=int, help="Batch size for training. (default=2)"
    )
    train_comput_group.add_argument(
        "--evaluation_steps",
        "-esteps",
        default=0,
        type=int,
        help="Fix the number of iterations to perform before computing an evaluation. Default will only "
        "perform one evaluation at the end of each epoch.",
    )

    train_data_group = train_parent_parser.add_argument_group(TRAIN_CATEGORIES["DATA"])
    train_data_group.add_argument(
        "--multi_cohort",
        help="Performs multi-cohort training. In this case, caps_directory and tsv_path must be paths to TSV files.",
        action="store_true",
        default=False,
    )
    train_data_group.add_argument(
        "--diagnoses",
        "-d",
        help="List of diagnoses that will be selected for training.",
        default=["AD", "CN"],
        nargs="+",
        type=str,
        choices=["AD", "BV", "CN", "MCI", "sMCI", "pMCI"],
    )
    train_data_group.add_argument(
        "--baseline",
        help="If provided, only the baseline sessions are used for training.",
        action="store_true",
        default=False,
    )
    train_data_group.add_argument(
        "--unnormalize",
        "-un",
        help="Disable default MinMaxNormalization.",
        action="store_true",
        default=False,
    )
    train_data_group.add_argument(
        "--data_augmentation",
        nargs="+",
        default=False,
        choices=["None", "Noise", "Erasing", "CropPad", "Smoothing"],
        help="Randomly applies transforms on the training set.",
    )
    train_data_group.add_argument(
        "--sampler",
        "-s",
        help="Sampler choice (random, or weighted for imbalanced datasets)",
        default="random",
        type=str,
        choices=["random", "weighted"],
    )

    train_cv_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["CROSS-VALIDATION"]
    )
    train_cv_group.add_argument(
        "--n_splits",
        help="If a value is given for k will load data of a k-fold CV. "
        "Default value (0) will load a single split.",
        type=int,
        default=0,
    )
    train_cv_group.add_argument(
        "--folds",
        help="Train the list of given folds. By default train all folds.",
        type=int,
        default=None,
        nargs="+",
    )

    train_optim_group = train_parent_parser.add_argument_group(
        TRAIN_CATEGORIES["OPTIMIZATION"]
    )
    train_optim_group.add_argument(
        "--epochs", help="Maximum number of epochs.", default=20, type=int
    )
    train_optim_group.add_argument(
        "--learning_rate",
        "-lr",
        help="Learning rate of the optimization.",
        default=1e-4,
        type=float,
    )
    train_optim_group.add_argument(
        "--weight_decay",
        "-wd",
        help="Weight decay value used in optimization.",
        default=1e-4,
        type=float,
    )
    train_optim_group.add_argument(
        "--dropout",
        help="rate of dropout that will be applied to dropout layers in CNN.",
        default=0,
        type=float,
    )
    train_optim_group.add_argument(
        "--patience",
        help="Number of epochs for early stopping patience.",
        type=int,
        default=0,
    )
    train_optim_group.add_argument(
        "--tolerance",
        help="Value for the early stopping tolerance.",
        type=float,
        default=0.0,
    )
    train_optim_group.add_argument(
        "--accumulation_steps",
        "-asteps",
        help="Accumulates gradients during the given number of iterations before performing the weight update "
        "in order to virtually increase the size of the batch.",
        default=1,
        type=int,
    )
    train_optim_group.add_argument(
        "--selection_metrics",
        help="Metrics used for the selection of the best models according to validation performance.",
        default=["loss"],
        nargs="+",
        type=str,
    )

    return train_parent_parser
