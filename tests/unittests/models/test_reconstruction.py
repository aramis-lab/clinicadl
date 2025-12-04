import torch
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.losses.config import MSELossConfig
from clinicadl.models import ReconstructionModel
from clinicadl.networks.config import AutoEncoderConfig
from clinicadl.optim.optimizers.config import AdamConfig

BATCH = Batch(
    [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 5, 5, 5)),
            label=0,
            participant="sub-0",
            session="ses-0",
        ),
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 5, 5, 5)),
            label=1,
            participant="sub-1",
            session="ses-1",
        ),
    ]
)


def test_ReconstructionModel():
    network = AutoEncoderConfig(
        in_shape=(1, 5, 5, 5), latent_size=1, conv_args={"channels": [1]}
    )
    loss = MSELossConfig()
    optimizer = AdamConfig()
    model = ReconstructionModel(network, loss, optimizer)

    # training step
    loss = model.forward_step(BATCH)
    assert loss.shape == ()
