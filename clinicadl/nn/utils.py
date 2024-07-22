from collections.abc import Iterable
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import _addindent


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""

    tmpstr = model.__class__.__name__ + " (\n"
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += "  (" + key + "): " + modstr
        if show_weights:
            tmpstr += ", weights={}".format(weights)
        if show_parameters:
            tmpstr += ", parameters={}".format(params)
        tmpstr += "\n"

    tmpstr = tmpstr + ")"
    return tmpstr


def multiply_list(L, factor):
    product = 1
    for x in L:
        product = product * x / factor
    return product


def compute_output_size(
    input_size: Union[torch.Size, Tuple], layer: nn.Module
) -> Tuple:
    """
    Computes the output size of a layer.

    Parameters
    ----------
    input_size : Union[torch.Size, Tuple]
        The unbatched input size (i.e. C, H, W(, D))
    layer : nn.Module
        The layer.

    Returns
    -------
    Tuple
        The unbatched size of the output.
    """
    input_ = torch.randn(input_size).unsqueeze(0)
    if isinstance(layer, nn.MaxUnpool3d) or isinstance(layer, nn.MaxUnpool2d):
        indices = torch.zeros_like(input_, dtype=int)
        print(indices)
        output = layer(input_, indices)
    else:
        output = layer(input_)
    if isinstance(layer, nn.MaxPool3d) or isinstance(layer, nn.MaxPool2d):
        if layer.return_indices:
            output = output[0]
    return tuple(output.shape[1:])
