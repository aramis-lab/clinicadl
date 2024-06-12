from pydantic import PositiveFloat, PositiveInt

from clinicadl.network.pythae import ModelConfig
from clinicadl.utils.enum import Normalization, ReconstructionLoss

from .ae_utils import AENetworks


class AEConfig(ModelConfig):
    network: AENetworks = AENetworks.AE_Conv5_FC3
    loss: ReconstructionLoss = ReconstructionLoss.MSELoss
    latent_space_size: PositiveInt = 128
    feature_size: PositiveInt = 1024
    n_conv: PositiveInt = 4
    io_layer_channels: PositiveInt = 8
    recons_weight: PositiveFloat = 1.0
    kl_weight: PositiveFloat = 1.0
    normalization: Normalization = Normalization.BATCH
