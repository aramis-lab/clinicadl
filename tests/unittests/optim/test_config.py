from clinicadl.optim import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(
        accumulation_steps=2, early_stopping={"patience": 7}, epochs=1
    )
    config.early_stopping.lower_bound = 0

    assert config.accumulation_steps == 2
    assert config.epochs == 1
    assert config.early_stopping.patience == 7
    assert config.early_stopping.lower_bound == 0
