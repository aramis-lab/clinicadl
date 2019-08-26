from .subject_level import Conv5_FC3, Conv5_FC3_mni
from .slice_level import Conv_4_FC_3
from .autoencoder import Decoder, initialize_other_autoencoder, transfer_learning


def create_model(model_name, gpu=False):

    try:
        model = eval(model_name)()
    except NameError:
        raise NotImplementedError(
            'The model wanted %s has not been implemented in the module subject_level.py' % options.model)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def create_decoder(model_name, gpu=False, transfer_learning=None, difference=0):
    model = create_model(model_name, gpu)
    decoder = Decoder(model)

    if transfer_learning is not None:
        decoder = initialize_other_autoencoder(decoder, transfer_learning, difference)

    return decoder


def load_model(model, checkpoint_dir, gpu, filename='model_best.pth.tar'):
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
    from os import path
    import torch

    if not path.exists(optimizer_path):
        raise ValueError('The optimizer was not found at path %s' % optimizer_path)
    print('Loading optimizer')
    optimizer_dict = torch.load(optimizer_path)
    name = optimizer_dict["name"]
    optimizer = eval("torch.optim." + name)(filter(lambda x: x.requires_grad, model.parameters()))
    optimizer.load_state_dict(optimizer_dict["optimizer"])
