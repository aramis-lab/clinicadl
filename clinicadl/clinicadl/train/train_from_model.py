"""
Retrain a model defined by a commandline.json file
"""

from copy import deepcopy
import warnings

from ..tools.deep_learning.iotools import read_json
from ..tools.deep_learning.models.random import find_evaluation_steps
from .train_autoencoder import train_autoencoder
from .train_singleCNN import train_single_cnn
from .train_multiCNN import train_multi_cnn


def set_options(options, new_options):
    from ..tools.deep_learning.iotools import computational_list
    arg_list = list(vars(new_options).keys())

    try:
        arg_list.remove("same_init")
        if new_options.same_init is not None:
            if new_options.same_init == "False":
                options.same_init = False
            else:
                options.same_init = new_options.same_init

    except ValueError:
        pass

    for arg in arg_list:
        new_value = getattr(new_options, arg)
        if new_value is not None or arg in computational_list:
            setattr(options, arg, new_value)

    return options


def retrain(new_options):

    options = deepcopy(new_options)
    options = read_json(options, read_computational=True)

    # Default behaviour reuse the same dataset as before
    options = set_options(options, new_options)

    if options.network_type == "autoencoder":
        train_autoencoder(options)
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
