"""
Script containing the iotools for model and optimizer creation / loading.
"""
from .subject_level import Conv5_FC3, Conv5_FC3_mni
from .patch_level import Conv4_FC3


def create_model(model_name, gpu=False):
    """
    Creates model object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :return: (Module) the model object
    """

    try:
        model = eval(model_name)()
    except NameError:
        raise NotImplementedError(
            'The model wanted %s has not been implemented.' % model_name)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def create_autoencoder(model_name, gpu=False, transfer_learning_path=None, difference=0):
    """
    Creates an autoencoder object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :param transfer_learning_path: (str) path to another pretrained autoencoder to perform transfer learning.
    :param difference: (int) difference of depth between the pretrained encoder and the new one.
    :return: (Module) the model object
    """
    from .autoencoder import AutoEncoder, initialize_other_autoencoder

    model = create_model(model_name, gpu)
    decoder = AutoEncoder(model)

    if transfer_learning_path is not None:
        decoder = initialize_other_autoencoder(decoder, transfer_learning_path, difference)

    return decoder


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
