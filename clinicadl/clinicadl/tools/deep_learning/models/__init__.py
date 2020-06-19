from .autoencoder import AutoEncoder, initialize_other_autoencoder, transfer_learning
from .iotools import load_model, load_optimizer, save_checkpoint
from .image_level import Conv5_FC3, Conv5_FC3_mni
from .patch_level import Conv4_FC3
from .slice_level import resnet18


def create_model(model_name, gpu=False, **kwargs):
    """
    Creates model object from the model_name.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param gpu: (bool) if True a gpu is used.
    :param init_state: (str) If 'same' will load
    :return: (Module) the model object
    """

    try:
        model = eval(model_name)(**kwargs)
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
    from os import path

    model = create_model(model_name, gpu)
    decoder = AutoEncoder(model)

    if transfer_learning_path is not None:
        if path.splitext(transfer_learning_path) != ".pth.tar":
            raise ValueError("The full path to the model must be given (filename included).")
        decoder = initialize_other_autoencoder(decoder, transfer_learning_path, difference)

    return decoder


def save_initialization(model_name, init_dir, init_state="random", autoencoder=False, **kwargs):
    """
    Saves the parameters initialization of a random model created from options and initial_shape.

    :param model_name: (str) the name of the model (corresponding exactly to the name of the class).
    :param init_dir: (str) path to the parent directory containing initialization state of the model.
    :param init_state: (str) If 'same' a state will be saved to be loaded at each training.
    :return: None
    """
    from os import path

    if init_state == 'same' and not path.exists(path.join(init_dir, "init.pth.tar")):
        model = create_model(model_name, **kwargs)
        if autoencoder:
            model = AutoEncoder(model)
        print(model.state_dict())
        init_dict = {"model": model.state_dict(),
                     "epoch": -1,
                     "valid_acc": None,
                     "valid_loss": None}
        save_checkpoint(init_dict, False, False, init_dir, filename="init.pth.tar")


def init_model(model_name, init_dir, init_state="random", autoencoder=False, gpu=False, **kwargs):

    model = create_model(model_name, gpu=gpu, **kwargs)
    if autoencoder:
        model = AutoEncoder(model)

    if init_state == 'same':
        model, _ = load_model(model, init_dir, gpu, filename="init.pth.tar")

    return model
