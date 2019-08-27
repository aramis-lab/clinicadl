

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
