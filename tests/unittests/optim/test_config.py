import pytest
from pydantic import ValidationError

from clinicadl.optim.config import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(
        accumulation_steps=2, num_epochs=1, evaluation_interval=2
    )

    assert config.accumulation_steps == 2
    assert config.num_epochs == 1
    assert config.evaluation_interval == 2

    with pytest.raises(ValidationError):
        OptimizationConfig(accumulation_steps=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(num_epochs=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(evaluation_interval=0)
