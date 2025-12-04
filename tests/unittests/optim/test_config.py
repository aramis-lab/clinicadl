import pytest
from pydantic import ValidationError

from clinicadl.optim.config import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(accumulation_steps=2, epochs=1, evaluation_steps=2)

    assert config.accumulation_steps == 2
    assert config.epochs == 1
    assert config.evaluation_steps == 2

    with pytest.raises(ValidationError):
        OptimizationConfig(accumulation_steps=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(epochs=0)
    with pytest.raises(ValidationError):
        OptimizationConfig(evaluation_steps=0)
