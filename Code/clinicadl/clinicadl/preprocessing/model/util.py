import torch
import os

# get models
from .resnet_qc    import *
from .squezenet_qc import *


def load_model(model, to_load):
    """
    load a previously trained model.
    """
    to_load = torch.load(to_load)

    # check parameters sizes
    model_params = set(model.state_dict().keys())
    to_load_params = set(to_load.state_dict().keys())

    assert model_params == to_load_params, (model_params - to_load_params, to_load_params - model_params)

    # copy saved parameters
    for k in model.state_dict().keys():
        if model.state_dict()[k].size() != to_load.state_dict()[k].size():
            raise Exception("Expected tensor {} of size {}, but got {}".format(
                k, model.state_dict()[k].size(),
                to_load.state_dict()[k].size()
            ))
        model.state_dict()[k].copy_(to_load.state_dict()[k])

def save_model(model, name, base):
    """
    Save the model.
    """
    if not os.path.exists(base):
        os.makedirs(base)
    
    path = os.path.join(base, '{}.pth'.format( name) )
    print('Saving the model to {} ...' .format( path))
    torch.save(model, path)


def get_qc_model(params, use_ref=False):
    if params.net=='r34':
        model=resnet_qc_34(pretrained=params.load is None, use_ref=use_ref)
    elif params.net=='r50':
        model=resnet_qc_50(pretrained=params.load is None, use_ref=use_ref)
    elif params.net=='r101':
        model=resnet_qc_101(pretrained=params.load is None, use_ref=use_ref)
    elif params.net=='r152':
        model=resnet_qc_152(pretrained=params.load is None, use_ref=use_ref)
    elif params.net=='sq101':
        model=squeezenet_qc(pretrained=params.load is None, use_ref=use_ref)
    else:
        model=resnet_qc_18(pretrained=params.load is None, use_ref=use_ref)
    
    if params.load is not None:
        load_model(model, params.load)

    return model