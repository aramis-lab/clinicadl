
def check_and_clean(d):
    import shutil
    import os

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def save_checkpoint(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_accuracy='best_acc', best_loss='best_loss'):
    import torch
    import os
    import shutil

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if accuracy_is_best:
        best_accuracy_path = os.path.join(checkpoint_dir, best_accuracy)
        if not os.path.exists(best_accuracy_path):
            os.makedirs(best_accuracy_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(best_accuracy_path, 'model_best.pth.tar'))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        if not os.path.exists(best_loss_path):
            os.makedirs(best_loss_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))


def commandline_to_json(commandline, model_type):
    """
    This is a function to write the python argparse object into a json file. This helps for DL when searching for hyperparameters
    :param commandline: a tuple contain the output of `parser.parse_known_args()`
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

    # save to json file
    json = json.dumps(commandline_arg_dic)
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
    from tools.deep_learning.data import MinMaxNormalization

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
