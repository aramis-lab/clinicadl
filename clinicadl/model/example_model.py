from clinicadl.losses.config import MSELossConfig
from clinicadl.networks.config import ResNetConfig
from clinicadl.optim.optimizers.config import AdamConfig

from .clinicadl_model import ClinicaDLModel

example_model = ClinicaDLModel(
    network=ResNetConfig(
        spatial_dims=3,
        in_channels=1,
        num_outputs=1,
    ),
    loss=MSELossConfig(),
    optimizer=AdamConfig(),
)
