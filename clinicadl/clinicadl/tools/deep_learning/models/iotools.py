# coding: utf8

"""
Script containing the iotools for model and optimizer serialization.
"""


def save_checkpoint(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_accuracy='best_balanced_accuracy', best_loss='best_loss'):
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


def load_model(model, checkpoint_dir, gpu, filename='model_best.pth.tar'):
    """
    Load the weights written in checkpoint_dir in the model object.

    :param model: (Module) CNN in which the weights will be loaded.
    :param checkpoint_dir: (str) path to the folder containing the parameters to loaded.
    :param gpu: (bool) if True a gpu is used.
    :param filename: (str) Name of the file containing the parameters to loaded.
    :return: (Module) the update model.
    """
    from copy import deepcopy
    import torch
    import os

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename), map_location="cpu")
    best_model.load_state_dict(param_dict['model'])

    if gpu:
        best_model = best_model.cuda()

    return best_model, param_dict['epoch']


def load_optimizer(optimizer_path, model):
    """
    Creates and load the state of an optimizer.

    :param optimizer_path: (str) path to the optimizer.
    :param model: (Module) model whom parameters will be optimized by the created optimizer.
    :return: optimizer initialized with specific state and linked to model parameters.
    """
    from os import path
    import torch

    if not path.exists(optimizer_path):
        raise ValueError('The optimizer was not found at path %s' % optimizer_path)
    print('Loading optimizer')
    optimizer_dict = torch.load(optimizer_path)
    name = optimizer_dict["name"]
    optimizer = eval("torch.optim." + name)(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer.load_state_dict(optimizer_dict["optimizer"])
