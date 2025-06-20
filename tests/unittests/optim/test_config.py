from clinicadl.optim.config import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(accumulation_steps=2, epochs=1)

    assert config.accumulation_steps == 2
    assert config.epochs == 1
