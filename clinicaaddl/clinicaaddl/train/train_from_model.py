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
    delattr(options, 'batch_size')
    options = read_json(options)

    # Adapt batch size with accumulation steps to match previous one
    if new_options.batch_size < options.batch_size:
        ratio = options.batch_size / new_options.batch_size
        if not ratio.is_integer():
            warnings.warn("The new batch size %i value is not a divisor of the previous one %i."
                          "The batch size for the training will be %i." %
                          (new_options.batch_size * int(options.accumulation_steps),
                           options.batch_size * int(options.accumulation_steps),
                           new_options.batch_size * int(ratio) * int(options.accumulation_steps)))
        options.accumulation_steps *= int(ratio)
        options.batch_size = new_options.batch_size

    elif new_options.batch_size > options.batch_size:
        if new_options.batch_size < options.batch_size * options.accumulation_steps:
            ratio = options.batch_size * options.accumulation_steps // new_options.batch_size
            warnings.warn("The previous batch size value was %i. "
                          "The new batch size value is %i." %
                          (options.batch_size * int(options.accumulation_steps),
                           new_options.batch_size * int(ratio)))
            options.accumulation_steps = ratio
            options.batch_size = new_options.batch_size

        else:
            warnings.warn("The previous batch size value was %i. "
                          "The new batch size value is %i." %
                          (options.batch_size * int(options.accumulation_steps),
                           new_options.batch_size))
            options.accumulation_steps = 1
            options.batch_size = new_options.batch_size

    # Update evaluation steps to match new accumulation steps value
    options.evaluation_steps = find_evaluation_steps(options.accumulation_steps, options.evaluation_steps)

    # Default behaviour reuse the same dataset as before
    options = set_options(options, new_options)

    if options.network_type == "autoencoder":
        train_autoencoder(options)
    elif options.network_type == "cnn":
        train_single_cnn(options)
    elif options.network_type == "multicnn":
        train_multi_cnn(options)
