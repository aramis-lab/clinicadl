
class Parameters:
    """ Class to define and initialize parameters used in traning CNN networks"""

    def __init__(self, tsv_path: str, output_dir: str, input_dir: str,
                 model: str):
        """
        Parameters:
        tsv_path: Path to the folder containing the tsv files of the
        population. To note, the column name should be participant_id,
        session_id and diagnosis.
        output_dir: Folder containing the results.
        input_dir: Path to the input folder with MRI in CAPS format.
        model: Neural network model.
        """
        self.tsv_path = tsv_path
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.model = model

    def write(self, 
            pretrained_path, 
            pretrained_difference,
            preprocessing: str = "linear",
            diagnoses: str = ["AD", "CN"],
            baseline: bool = False,
            minmaxnormalization: bool = False,
            sampler: str = "random",
            n_splits: int = 1,
            split: int = 0,
            accumulation_steps: int = 1,
            epochs: int = 20, 
            learning_rate: float = 1e-4,
            patience: int = 10, 
            tolerance: float = 0.05,
            add_sigmoid: bool = False,
            optimizer: str = "Adam",
            weight_decay: float = 0.1,
            gpu: bool = False,
            batch_size: int = 2,
            evaluation_steps: int = 1,
            num_workers: int = 1,
            transfer_learning_path: str = None,
            transfer_learning_autoencoder: str = None,
            selection: str = "best_acc",
            patch_size: int = 50,
            patch_stride: int = 50,
            hippocampus_roi: bool = False,
            selection_threshold: float = None,
            num_cnn: int = 36, 
            prepare_dl: bool = False,
            visualization: bool = False):
        """ 
        Optional parameters used for training CNN.
        pretrained_path: Path to a pretrained model (can be of different size).
        pretrained_difference: Difference of size between the pretrained 
                               autoencoder and the training.
        preprocessing: Type of preprocessing done. Choices: "linear" or "mni".
        diagnoses: Take all the subjects possible for autoencoder training.
        baseline: Use only the baseline if True.
        minmaxnormalization: Performs MinMaxNormalization.
        sampler: Sampler choice.
        n_splits: If a value is given will load data of a k-fold CV
        split: User can specify a chosen split.
        accumulation_steps: Accumulates gradients in order to increase the size
                            of the batch.
        epochs: Epochs through the data. (default=20).
        learning_rate: Learning rate of the optimization. (default=0.01).
        patience: Waiting time for early stopping.
        tolerance: Tolerance value for the early stopping.
        add_sigmoid: Ad sigmoid function at the end of the decoder.
        optimizer: Optimizer of choice for training. (default=Adam).
                   Choices=["SGD", "Adadelta", "Adam"].
        weight_decay: Weight decay of the optimizer. 
        gpu: GPU usage if True.
        batch_size: Batch size for training. (default=1)
        evaluation_steps: Fix the number of batches to use before validation
        num_workers:  Define the number of batch being loaded in parallel
        selection: Allow to choose which model of the experiment is loaded .
                   choices ["best_loss", "best_acc"]
        patch_size: The patch size extracted from the MRI.
        patch_stride: The stride for the patch extract window from the MRI
        hippocampus_roi: If train the model using only hippocampus ROI.
        selection_threshold: Threshold on the balanced accuracies to compute
                             the subject_level performance.
        num_cnn: How many CNNs we want to train in a patch-wise way.
                 By default, each patch is trained from all subjects for one CNN.
        prepare_dl: If True the outputs of preprocessing are used, else the
                    whole MRI is loaded. 
        """
        

        self.pretrained_path = pretrained_path
        self.pretrained_difference = pretrained_difference
        self.preprocessing = preprocessing
        self.diagnoses = diagnoses        
        self.baseline = baseline
        self.minmaxnormalization = minmaxnormalization
        self.sampler = sampler
        self.n_splits = n_splits
        self.split = split
        self.accumulation_steps = accumulation_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.tolerance = tolerance
        self.add_sigmoid = add_sigmoid
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gpu = gpu
        self.batch_size = batch_size
        self.evaluation_steps = evaluation_steps
        self.num_workers = num_workers
        self.transfer_learning_path = transfer_learning_path
        self.transfer_learning_autoencoder = transfer_learning_autoencoder
        self.selection = selection
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.hippocampus_roi = hippocampus_roi
        self.prepare_dl = prepare_dl
        self.visualization = visualization
        self.selection_threshold = selection_threshold

def check_and_clean(d):
    import shutil
    import os

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def commandline_to_json(commandline, model_type):
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
    if commandline_arg_dic['split'] is None:
        log_dir = os.path.join(output_dir, 'log_dir')
    else:
        log_dir = os.path.join(output_dir, 'log_dir', 'fold_' + str(commandline_arg_dic['split']))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if commandline_arg_dic['func'] is not None:
        del commandline_arg_dic['func']

    # save to json file
    json = json.dumps(commandline_arg_dic, skipkeys=True)
    print("Path of json file:", os.path.join(log_dir, "commandline_" + model_type + ".json"))
    f = open(os.path.join(log_dir, "commandline_" + model_type + ".json"), "w")
    f.write(json)
    f.close()


def read_json(options, model_type, json_path=None, test=False):
    """
    Read a json file to update python argparse Namespace.

    :param options: (argparse.Namespace) options of the model
    :return: options (args.Namespace) options of the model updated
    """
    import json
    from os import path

    evaluation_parameters = ["diagnosis_path", "input_dir", "diagnoses"]
    if json_path is None:
        json_path = path.join(options.model_path, 'log_dir', 'commandline_' + model_type + '.json')

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

    return options


def visualize_subject(decoder, dataloader, visualization_path, options, epoch=None, save_input=False, subject_index=0):
    from os import path, makedirs
    import nibabel as nib
    import numpy as np
    import torch
    from .data import MinMaxNormalization

    if not path.exists(visualization_path):
        makedirs(visualization_path)

    dataset = dataloader.dataset
    data = dataset[subject_index]
    image_path = data['image_path']
    nii_path, _ = path.splitext(image_path)
    nii_path += '.nii.gz'

    input_nii = nib.load(nii_path)
    input_np = input_nii.get_data().astype(float)
    np.nan_to_num(input_np, copy=False)
    input_pt = torch.from_numpy(input_np).unsqueeze(0).unsqueeze(0).float()
    if options.minmaxnormalization:
        transform = MinMaxNormalization()
        input_pt = transform(input_pt)

    if options.gpu:
        input_pt = input_pt.cuda()

    output_pt = decoder(input_pt)

    output_np = output_pt.detach().cpu().numpy()[0][0]
    output_nii = nib.Nifti1Image(output_np, affine=input_nii.affine)

    if save_input:
        nib.save(input_nii, path.join(visualization_path, 'input.nii'))

    if epoch is None:
        nib.save(output_nii, path.join(visualization_path, 'output.nii'))
    else:
        nib.save(output_nii, path.join(visualization_path, 'epoch-' + str(epoch) + '.nii'))


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
