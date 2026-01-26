import pytest
import torch
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.infer import SimpleInferer
from clinicadl.losses.config import BCEWithLogitsLossConfig
from clinicadl.models import SupervisedModel
from clinicadl.networks.config import ConvEncoderConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.utils.exceptions import CannotReadJsonFieldError

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
    inferer = SimpleInferer(output_name="my_output")
    model = SupervisedModel(network, loss, optimizer, inferer=inferer)
    optimizers = model.build_optimizers()

    # forward step
    loss = model.forward_step(BATCH)
    assert loss.shape == ()

    scaler = torch.amp.GradScaler(device="cpu")

    # backward step
    model.backward_step(loss, grad_scaler=scaler)
    assert next(iter(network.parameters())).grad is not None
    assert scaler._scale is not None

    # optimization step
    model.optimization_step(optimizers, scaler)
    assert 0 in optimizers["optimizer"].state_dict()["state"]

    # evaluation step
    out_batch = model.evaluation_step(BATCH)
    assert isinstance(out_batch, Batch)
    assert out_batch[0]["my_output"].shape == (1,)

    # prediction step
    out_batch = model.prediction_step(BATCH)
    assert isinstance(out_batch, Batch)
    assert out_batch[0]["my_output"].shape == (1,)

    # get losses
    assert model.get_loss_functions() == {"loss": model.loss}

    # write json
    model.to_json(tmp_path / "model.json")

    # from json
    with pytest.raises(
        CannotReadJsonFieldError,
        match=r"SupervisedModel cannot read the field\(s\) \['network'\] in.*",
    ):
        SupervisedModel.from_json(tmp_path / "model.json")
    new_model: SupervisedModel = SupervisedModel.from_json(
        tmp_path / "model.json",
        network=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(8, 1)),
    )
    assert isinstance(new_model, SupervisedModel)

    # summary
    summary_ = new_model.get_summary(input_data=BATCH)
    assert "Total params: 9" in summary_

    # network as config
    network = ConvEncoderConfig(spatial_dims=3, in_channels=1, channels=[2])
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamConfig()
    model = SupervisedModel(network, loss, optimizer)
    optimizers = model.build_optimizers()
