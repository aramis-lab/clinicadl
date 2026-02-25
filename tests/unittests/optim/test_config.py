import pytest
from pydantic import ValidationError

from clinicadl.optim.config import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(
        accumulation_steps=2,
        num_epochs=1,
        evaluation_interval=2,
        clip_grad_norm=1.1,
        grad_norm_type=-4.3,
        clip_grad_value=1.7,
    )

    assert config.accumulation_steps == 2
    assert config.num_epochs == 1
    assert config.evaluation_interval == 2
    assert config.clip_grad_norm == 1.1
    assert config.grad_norm_type == -4.3
    assert config.clip_grad_value == 1.7

    with pytest.raises(ValidationError):
        OptimizationConfig(accumulation_steps=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(num_epochs=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(evaluation_interval=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(clip_grad_norm=-0.1)
    with pytest.raises(ValidationError):
        OptimizationConfig(clip_grad_value=-0.1)
