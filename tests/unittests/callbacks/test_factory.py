import pytest

from clinicadl.callbacks.factory import ImplementedCallback, get_callback_from_dict
from clinicadl.callbacks.implemented import *
from clinicadl.optim.lr_schedulers.config import StepLRConfig
from clinicadl.utils.json import read_json

MANDATORY_ARGS = {
    "EarlyStoppingCallback": {"metric": "mse"},
    "LRSchedulerCallback": {"scheduler": StepLRConfig(step_size=1)},
}


@pytest.mark.parametrize(
    "callback",
    [globals()[name.value] for name in ImplementedCallback],
)
def test_callback_from_dict(callback, tmp_path):
    c = callback(**MANDATORY_ARGS.get(callback.__name__, {}))
    c.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    c = get_callback_from_dict(dict_)
    assert isinstance(c, callback)

    if callback is LRSchedulerCallback:
        assert isinstance(c.scheduler_config, StepLRConfig)
