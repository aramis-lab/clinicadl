import pytest
from pydantic import ValidationError

from clinicadl.optim.optimizer import OptimizerConfig


def test_OptimizerConfig():
    config = OptimizerConfig(
        optimizer="SGD",
        lr=1e-3,
        weight_decay={"param_0": 1e-3, "param_1": 1e-2},
        momentum={"param_1": 1e-1},
        lr_decay=1e-4,
    )
    assert config.optimizer == "SGD"
    assert config.lr == 1e-3
    assert config.weight_decay == {"param_0": 1e-3, "param_1": 1e-2}
    assert config.momentum == {"param_1": 1e-1}
    assert config.lr_decay == 1e-4
    assert config.alpha == "DefaultFromLibrary"
    assert sorted(config.get_all_groups()) == ["param_0", "param_1"]

    with pytest.raises(ValidationError):
        OptimizerConfig(betas={"params_0": (0.9, 1.01), "params_1": (0.9, 0.99)})
    with pytest.raises(ValidationError):
        OptimizerConfig(betas=0.9)
    with pytest.raises(ValidationError):
        OptimizerConfig(rho=1.01)
    with pytest.raises(ValidationError):
        OptimizerConfig(alpha=1.01)
    with pytest.raises(ValidationError):
        OptimizerConfig(dampening={"params_0": 0.1, "params_1": 2})
    with pytest.raises(ValueError):
        OptimizerConfig(optimizer="abc")
