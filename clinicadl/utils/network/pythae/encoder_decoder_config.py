from pydantic import BaseModel


class Encoder_Decoder_Config(BaseModel):
    input_size: list
    first_layer_channels: int = 32
    n_block_encoder: int = 5
    feature_size: int = 0
    latent_space_size: int = 256
    n_block_decoder: int = 5
    last_layer_channels: int = 32
    last_layer_conv: bool = False
    n_layer_per_block_encoder: int = 1
    n_layer_per_block_decoder: int = 1
    block_type: str = "conv"


def make_encoder_decoder_config(parameters):
    parameters["encoder_decoder_config"] = Encoder_Decoder_Config(**parameters)
    for key in parameters["encoder_decoder_config"].__fields__.keys():
        if key != "input_size":
            del parameters[key]
    return parameters
