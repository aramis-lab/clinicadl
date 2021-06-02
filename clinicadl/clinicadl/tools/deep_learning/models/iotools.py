# coding: utf8

"""
Script containing the iotools for model and optimizer serialization.
"""


def save_checkpoint(
    state,
    metrics_dict,
    checkpoint_dir,
    filename="checkpoint.pth.tar",
):
    """
    Update checkpoint and save the best model according to a dictionnary of metrics.
    If no metrics_dict is given, only the checkpoint is saved.

    Args:
        state: (dict) state of the training (model weights, epoch...)
        metrics_dict: (dict) key correspond to the name of the selection metric. The content is a boolean:
            - True if the saved model for this metric must be updated
            - False otherwise
        checkpoint_dir: (str) path to the checkpoint dir
        filename: (str) name of the checkpoint
    """
    import os
    import shutil
    from os.path import join

    import torch

    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(state, join(checkpoint_dir, filename))

    # Save model according to several metrics
    if metrics_dict is not None:
        for metric_name, metric_bool in metrics_dict.items():
            metric_path = join(checkpoint_dir, f"best_{metric_name}")
            if metric_bool:
                os.makedirs(metric_path, exist_ok=True)
                shutil.copyfile(
                    join(checkpoint_dir, filename),
                    join(metric_path, "model_best.pth.tar"),
                )


def load_model(model, checkpoint_dir, gpu, filename="model_best.pth.tar"):
    """
    Load the weights written in checkpoint_dir in the model object.

    :param model: (Module) CNN in which the weights will be loaded.
    :param checkpoint_dir: (str) path to the folder containing the parameters to loaded.
    :param gpu: (bool) if True a gpu is used.
    :param filename: (str) Name of the file containing the parameters to loaded.
    :return: (Module) the update model.
    """
    import os
    from copy import deepcopy

    import torch

    best_model = deepcopy(model)
    param_dict = torch.load(os.path.join(checkpoint_dir, filename), map_location="cpu")
    best_model.load_state_dict(param_dict["model"])

    if gpu:
        best_model = best_model.cuda()

    return best_model, param_dict["epoch"]


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
        raise ValueError("The optimizer was not found at path %s" % optimizer_path)
    print("Loading optimizer")
    optimizer_dict = torch.load(optimizer_path)
    name = optimizer_dict["name"]
    optimizer = getattr(torch.optim, name)(
        filter(lambda x: x.requires_grad, model.parameters())
    )
    optimizer.load_state_dict(optimizer_dict["optimizer"])

    return optimizer
