from clinicadl.callbacks.handler import _CallbacksHandler
from clinicadl.IO.maps.maps import Maps
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.optim.config import OptimizationConfig
from clinicadl.train.trainer import Trainer
from clinicadl.utils.computational.config import ComputationalConfig

from ..resources.objects import (
    CALLBACKS,
    COMP,
    METRICS,
    MODEL,
    OPTIM,
    SPLIT,
)


def assert_equal(obj1, obj2):
    """
    Assert that two objects are equal, ignoring private attributes and
    attributes starting with an underscore.
    """

    def _normalize(value):
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items() if not k.startswith("_")}
        elif isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]  # On convertit tout en liste
        elif hasattr(value, "__dict__"):
            return _normalize(value.__dict__)
        else:
            return value

    norm1 = _normalize(obj1)
    norm2 = _normalize(obj2)

    assert norm1 == norm2, f"Objects differ:\n{norm1}\n≠\n{norm2}"


def test_training_from_json():
    trainer = Trainer(
        maps_path="maps_tests",
        model=MODEL,
        optim_config=OPTIM,
        comp_config=COMP,
        callbacks=CALLBACKS,
        metrics=METRICS,
        _overwrite=True,
    )

    maps = Maps("maps_tests")
    assert maps.path == trainer.maps.path

    model = ClinicaDLModel.from_json(maps.model_json)
    assert_equal(model.network, trainer.model.network)
    # assert_equal(model.loss, trainer.model.loss)
    assert_equal(model.optimizer, trainer.model.optimizer)

    comp_config = ComputationalConfig.from_json(maps.training.computational_json)
    assert_equal(comp_config, trainer.config.comp)

    optim_config = OptimizationConfig.from_json(maps.training.optimization_json)
    assert_equal(optim_config, trainer.config.optim)

    callbacks = _CallbacksHandler.from_json(maps.training.callbacks_json)
    metrics = MetricsHandler.from_json(maps.training.metrics_json, mae=METRICS["mae"])

    new_trainer = Trainer(
        maps_path="maps_tests_bis",
        model=model,
        optim_config=optim_config,
        comp_config=comp_config,
        callbacks=callbacks,
        metrics=metrics.metrics,
        _overwrite=True,
    )
    assert_equal(new_trainer.model, trainer.model)
    assert_equal(new_trainer.config.optim, trainer.config.optim)
    assert_equal(new_trainer.config.comp, trainer.config.comp)

    for cb1, cb2 in zip(new_trainer.callbacks.callbacks, trainer.callbacks.callbacks):
        assert isinstance(
            cb1, type(cb2)
        ), f"Callback types differ: {type(cb1)} vs {type(cb2)}"
        assert_equal(cb1, cb2)

    assert_equal(new_trainer.metrics, trainer.metrics)
