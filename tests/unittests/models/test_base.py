import json

import pytest

from clinicadl.models import ClinicaDLModel
from clinicadl.utils.exceptions import NotInterpretableJson


class CustomModel(ClinicaDLModel):
    def forward_step(self):
        pass

    def optimization_step(self):
        pass

    def evaluation_step(self):
        pass

    def get_loss_functions(self):
        pass

    def get_optimizers(self):
        pass

    def to(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def write_architecture_log(self):
        pass


def test_write_and_read_json(tmp_path):
    model = CustomModel()
    model.write_json(tmp_path / "model.json")
    with open(tmp_path / "model.json") as f:
        dict_ = json.load(f)
    assert dict_ == "Custom model passed by the user: CustomModel"

    with pytest.raises(NotInterpretableJson, match="ClinicaDLModel cannot read*"):
        ClinicaDLModel.from_json(tmp_path / "model.json")
