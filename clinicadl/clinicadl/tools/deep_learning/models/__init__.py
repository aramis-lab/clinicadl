from .subject_level import Conv5_FC3, Conv5_FC3_mni
from .slice_level import Conv_4_FC_3
from .autoencoder import AutoEncoder, initialize_other_autoencoder, transfer_learning
from .iotools import load_model, load_optimizer


def create_model(model_name, gpu=False):

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


def create_autoencoder(model_name, gpu=False, transfer_learning=None, difference=0):
    model = create_model(model_name, gpu)
    decoder = AutoEncoder(model)

    if transfer_learning is not None:
        decoder = initialize_other_autoencoder(decoder, transfer_learning, difference)

    return decoder
