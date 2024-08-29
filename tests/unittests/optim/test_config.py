from clinicadl.optim import OptimizationConfig


def test_OptimizationConfig():
    config = OptimizationConfig(
        **{
            "accumulation_steps": 2,
            "early_stopping": {"patience": 7},
        }
    )
    config.early_stopping.lower_bound = 0

    assert config.accumulation_steps == 2
    assert config.epochs == 20
    assert config.early_stopping.patience == 7
    assert config.early_stopping.lower_bound == 0
    assert config.early_stopping.check_finite
