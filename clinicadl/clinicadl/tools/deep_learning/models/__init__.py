from .autoencoder import AutoEncoder, initialize_other_autoencoder, transfer_learning
from .iotools import load_model, load_optimizer, save_checkpoint
from .image_level import Conv5_FC3, Conv5_FC3_mni
from .patch_level import Conv4_FC3
from .slice_level import resnet18


def create_model(model_name, gpu=False, ae_from_model=False, **kwargs):
    """
    Creates model object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :param ae_from_model: (bool) if True an autoencoder is built from the model.
    :return: (Module) the model object
    """

    try:
        model = eval(model_name)(**kwargs)
    except NameError:
        raise NotImplementedError(
            'The model wanted %s has not been implemented.' % model_name)

    if ae_from_model:
        model = AutoEncoder(model)

    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
