import pytest
from torch.nn import MSELoss

from clinicadl.losses.config import MSELossConfig
from clinicadl.models import ReconstructionModel, SupervisedModel
from clinicadl.models.factory import get_model_from_json
from clinicadl.networks.config import MLPConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.utils.exceptions import CannotReadJsonFieldError


@pytest.mark.parametrize(
    "model",
    [
        SupervisedModel,
        ReconstructionModel,
    ],
)
def test_get_model_from_json(tmp_path, model):
    m = model(
        loss=MSELossConfig(),
        network=MLPConfig(num_inputs=1, num_outputs=1, hidden_dims=[]),
        optimizer=AdamConfig(),
    )

    # write json
    m.to_json(tmp_path / "model.json")
    m = get_model_from_json(tmp_path / "model.json")
    assert isinstance(m, model)

    if model is SupervisedModel:
        assert m.network.config.num_inputs == 1

        m = model(
            loss=MSELoss(),
            network=MLPConfig(num_inputs=1, num_outputs=1, hidden_dims=[]),
            optimizer=AdamConfig(),
        )
        m.to_json(tmp_path / "model.json", overwrite=True)
        with pytest.raises(
            CannotReadJsonFieldError,
            match="SupervisedModel cannot read the field\\(s\\) \\['loss'\\] in .*\nPlease pass this field via kwargs",
        ):
            m = get_model_from_json(tmp_path / "model.json")

        m = SupervisedModel.from_json(tmp_path / "model.json", loss=MSELoss())
        assert isinstance(m, SupervisedModel)
