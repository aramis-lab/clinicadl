# coding: utf8


class Parameters:
    """ Class to define and initialize parameters used in training CNN networks"""

    def __init__(self, mode: str, tsv_path: str, output_dir: str, input_dir: str,
                 preprocessing: str, model: str):
        """
        Parameters:
        mode: type if input used by the network (image, patch, roi, slice)
        tsv_path: Path to the folder containing the tsv files of the
        population. To note, the column name should be participant_id,
        session_id and diagnosis.
        output_dir: Folder containing the results.
        input_dir: Path to the input folder with MRI in CAPS format.
        preprocessing: Type of preprocessing done. Choices: "t1-linear" or "t1-extensive".
        model: Neural network model.
        """
        self.mode = mode
        self.tsv_path = tsv_path
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.preprocessing = preprocessing
        self.model = model

    def write(
            self,
            transfer_learning_difference: int = 0,
            diagnoses: str = ["AD", "CN"],
            baseline: bool = False,
            minmaxnormalization: bool = False,
            n_splits: int = 1,
            split: int = 0,
            accumulation_steps: int = 1,
            epochs: int = 20,
            learning_rate: float = 1e-4,
            patience: int = 10,
            tolerance: float = 0.05,
            optimizer: str = "Adam",
            weight_decay: float = 1e-4,
            dropout: float = 0,
            gpu: bool = False,
            batch_size: int = 12,
            evaluation_steps: int = 0,
            num_workers: int = 1,
            transfer_learning_path: str = None,
            transfer_learning_selection: str = "best_acc",
            patch_size: int = 50,
            stride_size: int = 50,
            hippocampus_roi: bool = False,
            selection_threshold: float = 0.0,
            mri_plane: int = 0,
            discarded_slices: int = 20,
            prepare_dl: bool = False,
            visualization: bool = False
    ):
        """
        Optional parameters used for training CNN.
        transfer_learning_difference: Difference of size between the pretrained
                               autoencoder and the training.
        diagnoses: Take all the subjects possible for autoencoder training.
        baseline: Use only the baseline if True.
        minmaxnormalization: Performs MinMaxNormalization.
        n_splits: If a value is given will load data of a k-fold CV
        split: User can specify a chosen split.
        accumulation_steps: Accumulates gradients in order to increase the size
                            of the batch.
        epochs: Epochs through the data. (default=20).
        learning_rate: Learning rate of the optimization. (default=0.01).
        patience: Waiting time for early stopping.
        tolerance: Tolerance value for the early stopping.
        optimizer: Optimizer of choice for training. (default=Adam).
                   Choices=["SGD", "Adadelta", "Adam"].
        weight_decay: Weight decay of the optimizer.
        gpu: GPU usage if True.
        batch_size: Batch size for training. (default=1)
        evaluation_steps: Fix the number of batches to use before validation
        num_workers:  Define the number of batch being loaded in parallel
        transfer_learning_selection: Allow to choose from which model the weights are transferred.
                    Choices ["best_loss", "best_acc"]
        patch_size: The patch size extracted from the MRI.
        stride_size: The stride for the patch extract window from the MRI
        hippocampus_roi: If train the model using only hippocampus ROI.
        selection_threshold: Threshold on the balanced accuracies to compute
                             the subject-level performance.
        mri_plane: Which coordinate axis to take for slicing the MRI.
                   0 is for sagittal,
                   1 is for coronal and
                   2 is for axial direction
        prepare_dl: If True the outputs of preprocessing are used, else the
                    whole MRI is loaded.
                                     initialize corresponding models.
        """

        self.transfer_learning_difference = transfer_learning_difference
        self.diagnoses = diagnoses
        self.baseline = baseline
        self.minmaxnormalization = minmaxnormalization
        self.n_splits = n_splits
        self.split = split
        self.accumulation_steps = accumulation_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tolerance = tolerance
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.gpu = gpu
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.num_workers = num_workers
        self.transfer_learning_path = transfer_learning_path
        self.transfer_learning_selection = transfer_learning_selection
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.hippocampus_roi = hippocampus_roi
        self.mri_plane = mri_plane
        self.discarded_slices = discarded_slices
        self.prepare_dl = prepare_dl
        self.visualization = visualization
        self.selection_threshold = selection_threshold


def check_and_clean(d):
    import shutil
    import os

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def commandline_to_json(commandline):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters

    :param commandline: a tuple contain the output of
                        `parser.parse_known_args()`

    :return:
    """
    import json
    import os

    commandline_arg_dic = vars(commandline[0])
    commandline_arg_dic['unknown_arg'] = commandline[1]

    output_dir = commandline_arg_dic['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # remove these entries from the commandline log file
    if 'func' in commandline_arg_dic:
        del commandline_arg_dic['func']

    if 'caps_dir' in commandline_arg_dic:
        del commandline_arg_dic['caps_dir']

    if 'tsv_path' in commandline_arg_dic:
        del commandline_arg_dic['tsv_path']

    if 'output_dir' in commandline_arg_dic:
        del commandline_arg_dic['output_dir']

    # save to json file
    json = json.dumps(commandline_arg_dic, skipkeys=True, indent=4)
    print("Path of json file:", os.path.join(output_dir, "commandline.json"))
    f = open(os.path.join(output_dir, "commandline.json"), "w")
    f.write(json)
    f.close()


def read_json(options, json_path=None, test=False):
    """
    Read a json file to update python argparse Namespace.

    Args:
        options: (argparse.Namespace) options of the model.
        json_path: (str) If given path to the json file, else found with options.model_path.
        test: (bool) If given the reader will ignore some options specific to data.
    Returns:
        options (args.Namespace) options of the model updated
    """
    import json
    from os import path
    from ...cli import set_default_dropout

    evaluation_parameters = ["diagnosis_path", "input_dir", "diagnoses"]
    prep_compatibility_dict = {"mni": "t1-extensive", "linear": "t1-linear"}
    if json_path is None:
        json_path = path.join(options.model_path, 'commandline.json')

    with open(json_path, "r") as f:
        json_data = json.load(f)

    for key, item in json_data.items():
        # We do not change computational options
        if key in ['gpu', 'num_workers', 'num_threads']:
            pass
        # If used for evaluation, some parameters were already given
        if test and key in evaluation_parameters:
            pass
        else:
            setattr(options, key, item)

    # Retro-compatibility with runs of previous versions
    if not hasattr(options, "model"):
        options.model = options.network
        del options.network

    if not hasattr(options, 'dropout'):
        options.dropout = None
    set_default_dropout(options)

    if not hasattr(options, 'discarded_sliced'):
        options.discarded_slices = 20

    if options.preprocessing in prep_compatibility_dict.keys():
        options.preprocessing = prep_compatibility_dict[options.preprocessing]

    if hasattr(options, 'mri_plane'):
        options.slice_direction = options.mri_plane
        del options.mri_plane

    if hasattr(options, "hippocampus_roi"):
        if options.hippocampus_roi:
            options.mode = "roi"
            del options.hippocampus_roi

    if hasattr(options, "pretrained_path"):
        options.transfer_learning_path = options.pretrained_path
        del options.pretrained_path

    if hasattr(options, "pretrained_difference"):
        options.transfer_learning_difference = options.pretrained_difference
        del options.pretrained_difference

    if hasattr(options, 'slice_direction'):
        options.mri_plane = options.slice_direction

    if hasattr(options, 'patch_stride'):
        options.stride_size = options.patch_stride

    if hasattr(options, 'use_gpu'):
        options.use_cpu = not options.use_gpu

    if hasattr(options, 'use_extracted_patches'):
        options.prepare_dl = not options.use_extracted_patches

    if options.mode == "subject":
        options.mode = "image"
    if options.mode == "slice" and not hasattr(options, "mode_task"):
        options.mode_task = "cnn"
    if options.mode == "patch" and hasattr(options, "network_type"):
        if options.network_type == "multi":
            options.mode_task = "multicnn"
        del options.network_type

    if not hasattr(options, "mode_task"):
        if hasattr(options, "train_autoencoder"):
            options.mode_task = "autoencoder"
        else:
            options.mode_task = "cnn"

    if hasattr(options, "use_cpu"):
        options.gpu = not options.use_cpu

    if hasattr(options, "unnormalize"):
        options.minmaxnormalization = not options.unnormalize

    if hasattr(options, "selection"):
        options.transfer_learning_selection = options.selection

    if hasattr(options, "use_extracted_slices"):
        options.prepare_dl = options.use_extracted_slices
    if hasattr(options, "use_extracted_patches"):
        options.prepare_dl = options.use_extracted_patches
    if hasattr(options, "use_extracted_roi"):
        options.prepare_dl = options.use_extracted_roi

    return options


def memReport():
    import gc
    import torch

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil
    import os

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
