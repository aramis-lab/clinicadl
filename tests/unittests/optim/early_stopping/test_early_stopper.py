import numpy as np

from clinicadl.optim.early_stopping import EarlyStopping, EarlyStoppingConfig


def test_EarlyStopping():
    config = EarlyStoppingConfig(
        patience=2, min_delta=0.1, mode="max", check_finite=True
    )
    early_stopping = EarlyStopping(config)
    assert early_stopping.step(np.nan)
    assert early_stopping.step(np.inf)
    assert not early_stopping.step(1.0)
    assert not early_stopping.step(0.8)
    assert not early_stopping.step(1.11)
    assert not early_stopping.step(1.0)
    assert early_stopping.step(0.8)

    config = EarlyStoppingConfig(
        patience=3, min_delta=0.1, mode="min", upper_bound=10.0, lower_bound=-10.0
    )
    early_stopping = EarlyStopping(config)
    assert early_stopping.step(11.0)
    assert early_stopping.step(-11.0)
    assert not early_stopping.step(1.0)
    assert not early_stopping.step(0.98)
    assert not early_stopping.step(0.95)
    assert early_stopping.step(0.92)

    config = EarlyStoppingConfig(patience=1, min_delta=0.0, mode="min")
    early_stopping = EarlyStopping(config)
    assert not early_stopping.step(-1.0)
    assert not early_stopping.step(-1.1)
    assert early_stopping.step(-1.1)

    config = EarlyStoppingConfig(patience=1, min_delta=0.0, mode="max")
    early_stopping = EarlyStopping(config)
    assert not early_stopping.step(-1.0)
    assert not early_stopping.step(-0.9)
    assert not early_stopping.step(-0.8)
    assert early_stopping.step(-0.81)

    config = EarlyStoppingConfig(patience=None, check_finite=True, mode="min")
    early_stopping = EarlyStopping(config)
    for k in range(10):
        assert not early_stopping.step(10 * k)
    assert early_stopping.step(np.nan)
