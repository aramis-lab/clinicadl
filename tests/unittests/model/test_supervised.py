import json

import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.losses.config import BCEWithLogitsLossConfig
from clinicadl.model import SupervisedModel
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

    # training step
    loss = model.training_step(BATCH)
    assert loss.shape == ()

    # evaluation step
    out_batch = model.evaluation_step(BATCH)
    assert isinstance(out_batch, Batch)
    assert out_batch[0]["output"].shape == (1,)

    # get optimizers
    opt = model.get_optimizers()
    assert isinstance(opt, torch.optim.Adam)

    # fake an optimizer step
    loss.backward()
    opt.step()

    # eval
    model.eval()
    assert not network.training

    # train
    model.train()
    assert network.training

    # write json
    model.write_json(tmp_path / "model.json")
    with open(tmp_path / "model.json", "r") as f:
        dict_ = json.load(f)
    assert dict_ == {
        "network": "Custom network passed by the user: 'Sequential'",
        "loss": {
            "name": "BCEWithLogitsLoss",
            "weight": None,
            "reduction": "mean",
            "pos_weight": None,
        },
        "optimizer": {
            "name": "Adam",
            "freeze": None,
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": None,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        },
    }

    # write checkpoint
    model.save_checkpoint(tmp_path / "weights.pt", only_network_weights=True)
    model.save_checkpoint(tmp_path / "model_state.pt", only_network_weights=False)

    # from json
    with pytest.raises(
        ValueError,
        match=r"Custom network found for in .*\. ClinicaDL can't read custom network, so pass it to 'from_json' via network=<your-custom-network>",
    ):
        SupervisedModel.from_json(tmp_path / "model.json")
    new_model = SupervisedModel.from_json(
        tmp_path / "model.json",
        network=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1)),
    )

    # read checkpoint
    new_model.load_checkpoint(
        tmp_path / "weights.pt", device="cpu", only_network_weights=True
    )
    torch.testing.assert_close(
        next(iter(network.parameters())), next(iter(new_model.network.parameters()))
    )

    new_model.load_checkpoint(
        tmp_path / "model_state.pt", device="cpu", only_network_weights=False
    )
    torch.testing.assert_close(
        next(iter(network.parameters())), next(iter(new_model.network.parameters()))
    )
    torch.testing.assert_close(
        opt.state_dict()["state"][0]["exp_avg"],
        new_model.optimizer.state_dict()["state"][0]["exp_avg"],
    )

    # architecture
    new_model.write_architecture_log(tmp_path / "architecture.log")
    with open(tmp_path / "architecture.log", "r", encoding="utf-8") as f:
        content = f.read()
    assert (
        content
        == "Sequential(\n  (0): Flatten(start_dim=1, end_dim=-1)\n  (1): Linear(in_features=8, out_features=1, bias=True)\n)\n"
    )


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
    model.to(device="cuda", memory_format=torch.channels_last_3d, dtype=torch.float16)
    param = next(iter(model.network.parameters()))
    param.stride() == (27, 1, 9, 3, 1)
    param.dtype == torch.float16
    param.device == torch.device("cuda:0")
