import pytest

from clinicadl.optim.optimizers.config import *


@pytest.mark.parametrize(
    "name,config",
    [
        ("Adadelta", AdadeltaConfig),
        ("Adagrad", AdagradConfig),
        ("Adam", AdamConfig),
        ("RMSprop", RMSpropConfig),
        ("SGD", SGDConfig),
    ],
)
def test_get_optimizer_config(name, config):
    c = get_optimizer_config(name)
    assert c.name == name
    assert isinstance(c, config)

    if name == "Adadelta":
        config = get_optimizer_config("Adadelta", lr=0.5, capturable=True)
        assert config.name == "Adadelta"
        assert config.lr == 0.5
        assert config.capturable

        with pytest.raises(ValueError):
            get_optimizer_config("abc")
