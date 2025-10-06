import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.losses.config import BCEWithLogitsLossConfig
from clinicadl.models import SupervisedModel
from clinicadl.networks.config import ConvEncoderConfig
from clinicadl.optim.optimizers.config import AdamConfig

BATCH = Batch(
    [
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
            label=0,
            participant="sub-0",
            session="ses-0",
        ),
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
            label=1,
            participant="sub-1",
            session="ses-1",
        ),
    ]
)


def test_SupervisedModel(tmp_path):
    network = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1))
    loss = BCEWithLogitsLossConfig()
    optimizer = AdamConfig()
    model = SupervisedModel(network, loss, optimizer)

    # forward step
    loss = model.forward_step(BATCH)
    assert loss.shape == ()

    scaler = torch.amp.GradScaler(device="cpu")

    # backward step
    model.backward_step(loss, grad_scaler=scaler)
    assert next(iter(network.parameters())).grad is not None
    assert scaler._scale is not None

    # optimization step
    model.optimization_step()
    assert 0 in model.optimizer.state_dict()["state"]

    # evaluation step
    out_batch = model.evaluation_step(BATCH)
    assert isinstance(out_batch, Batch)
    assert out_batch[0]["output"].shape == (1,)

    # get losses
    assert model.get_loss_functions() == {"loss": model.loss}

    # get optimizers
    assert model.get_optimizers() == {"optimizer": model.optimizer}

    # eval
    model.eval()
    assert not network.training

    # train
    model.train()
    assert network.training

    # state_dict
    state_dict = model.state_dict()

    # write reaad json
    model.to_json(tmp_path / "model.json")
    new_model = SupervisedModel.from_json(tmp_path / "model.json", network=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1)))

    # read checkpoint
    new_model.load_state_dict(state_dict)
    torch.testing.assert_close(
        next(iter(network.parameters())), next(iter(new_model.network.parameters()))
    )
    torch.testing.assert_close(
        model.optimizer.state_dict()["state"][0]["exp_avg"],
        new_model.optimizer.state_dict()["state"][0]["exp_avg"],
    )

    # architecture
    archi = new_model.get_architecture()
    assert (
        archi
        == "Sequential(\n  (0): Flatten(start_dim=1, end_dim=-1)\n  (1): Linear(in_features=8, out_features=1, bias=True)\n)"
    )

    # summary
    summary_ = new_model.get_summary(input_data=torch.randn(1, 2, 2, 2))
    assert "Total params: 9" in summary_


@pytest.mark.gpu
def test_gpu():
    network = ConvEncoderConfig(spatial_dims=3, in_channels=1, channels=[2])
    loss = BCEWithLogitsLossConfig()
    optimizer = AdamConfig()
    model = SupervisedModel(network, loss, optimizer)

    param = next(iter(model.network.parameters()))
    param.stride() == (27, 27, 9, 3, 1)
    param.dtype == torch.float32
    param.device == torch.device("cpu")
    model.to(
        device="cuda",
        memory_format=torch.channels_last_3d,
        dtype=torch.float16,
        non_blocking=True,
    )
    param = next(iter(model.network.parameters()))
    param.stride() == (27, 1, 9, 3, 1)
    param.dtype == torch.float16
    param.device == torch.device("cuda:0")
