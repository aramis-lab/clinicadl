import os
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn
import torchio as tio

from clinicadl.callbacks import Callback, CallbacksHandler
from clinicadl.data.dataloader import Batch, CollateFn, DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.io import Maps
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import MetricConfig, MSEMetricConfig
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.utils.exceptions import CannotReadJsonError

MAPS_PATH = Path(__file__).parents[1] / "resources" / "maps_example"
CAPS_PATH = Path(__file__).parents[1] / "resources" / "caps_example"


def _add_maps_to_trainer(trainer: Trainer, tmp_path: Path) -> Maps:
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    maps = Maps(tmp_path)
    maps.read()
    trainer._maps = maps

    return maps


def _setup_dataloader():
    from clinicadl.data.dataloader import Batch
    from clinicadl.data.structures import DataPoint

    batch = Batch(
        DataPoint(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
            participant=str(i),
            session=str(i),
            label=i,
        )
        for i in range(2)
    )

    loader = MagicMock()
    loader.__len__.return_value = 2
    loader.__iter__.return_value = [batch, batch]

    return loader


def _setup_real_model():
    from clinicadl.models import SupervisedModel
    from clinicadl.optim.optimizers.config import AdamConfig

    model = SupervisedModel(
        network=nn.Sequential(nn.Conv3d(1, 2, 3), nn.Flatten(), nn.Linear(2, 1)),
        loss=nn.MSELoss(),
        optimizer=AdamConfig(),
    )

    return model


@pytest.fixture()
def trainer(tmp_path) -> Trainer:
    optim = Mock()
    optim.num_epochs = 5
    cb = Callback()
    cb.on_exception = Mock()
    trainer = Trainer(
        maps=tmp_path / "maps",
        model=Mock(),
        metrics={},
        optimization=optim,
        callbacks=(list_callbacks := [Callback(), Mock(spec=Callback)]),
    )
    trainer._callbacks._all_callbacks = list_callbacks  # not to call all the callbacks

    return trainer


@pytest.fixture()
def custom_metric() -> MetricConfig:
    class CustomMetricConfig(MetricConfig):
        @staticmethod
        def optimum():
            return "max"

        @classmethod
        def _get_class(cls):
            return Mock()

    return CustomMetricConfig()


class TestSideMethods:
    def test_init(self, tmp_path, custom_metric):
        model = Mock()
        optimization = Mock()
        cb = Callback()
        cb.on_trainer_init = Mock()

        trainer = Trainer(
            maps=tmp_path / "maps",
            model=model,
            metrics={"my_metric": custom_metric},
            optimization=optimization,
            callbacks=[cb],
        )
        assert trainer.maps.path == tmp_path / "maps"
        assert trainer.model is model
        assert list(trainer.metrics.metrics.keys()) == ["my_metric"]
        assert trainer.optimization is optimization
        assert trainer.callbacks.callbacks[-3] is cb
        trainer.callbacks.callbacks[-3].on_trainer_init.assert_called_once_with(
            model=model,
            maps=trainer.maps,
            state=trainer.state,
            metrics=trainer.metrics,
            optimization=optimization,
            callbacks=trainer.callbacks,
        )
        assert trainer.state.split_idx is None

        metrics = MetricsHandler(my_metric=custom_metric)
        callbacks = CallbacksHandler([cb])
        trainer = Trainer(
            maps=Maps(tmp_path / "maps"),
            model=model,
            metrics=metrics,
            callbacks=callbacks,
            overwrite=True,
        )
        assert trainer.maps.path == tmp_path / "maps"
        assert trainer.callbacks is callbacks
        assert trainer.metrics is metrics
        assert trainer.optimization.num_epochs == 10

        with pytest.raises(FileExistsError):
            Trainer(tmp_path / "maps", model=model)

        loss = Mock()
        loss.reduction = "mean"
        model.get_loss_functions.return_value = {"loss": loss}
        trainer = Trainer(tmp_path / "maps", model=model, overwrite=True)
        model.get_loss_functions.assert_called_once()
        assert len(trainer.callbacks.config.callbacks) == 0
        assert list(trainer.metrics.metrics.keys()) == ["loss"]

    def test_add_metrics(self, trainer: Trainer, tmp_path, custom_metric):
        maps = _add_maps_to_trainer(trainer, tmp_path)
        trainer.add_metrics(my_metric=custom_metric)
        assert "my_metric" in trainer.metrics.metrics
        f = maps.open_file(maps.metrics_json)
        assert f["metrics"]["my_metric"]["name"] == "CustomMetric"

    def test_add_callbacks(self, trainer: Trainer, tmp_path):
        class CustomCallback(Callback):
            pass

        maps = _add_maps_to_trainer(trainer, tmp_path)

        assert len(trainer.callbacks.callbacks) == 6
        trainer.add_callbacks([CustomCallback()])
        assert isinstance(trainer.callbacks.callbacks[-3], CustomCallback)
        f = maps.open_file(maps.callbacks_json)
        assert len(f["callbacks"]) == 3

    def test_reset_train(self, trainer: Trainer):
        split = Mock()
        split.index = 1
        metrics = Mock()

        trainer._reset_train(split, metrics)

        assert trainer.state.split_idx == 1
        assert trainer.state.num_epochs == 5
        assert trainer.state.called == "train"
        trainer.model.reset.assert_called_once()
        split.train_dataset.train.assert_called_once()
        split.val_dataset.eval.assert_called_once()
        metrics.reset.assert_called_once_with(reset_df=True)

    def test_reset_epoch(self, trainer: Trainer):
        train_loader = MagicMock()
        train_loader.__len__.return_value = 10

        trainer._reset_epoch(2, train_loader)

        assert trainer.state.current_epoch == 2
        assert trainer.state.num_train_batches == 10
        trainer.model.train.assert_called_once()
        train_loader.set_epoch.assert_called_once_with(2)

    def test_reset_validation(self, trainer: Trainer):
        val_loader = MagicMock()
        val_loader.__len__.return_value = 9
        metrics = Mock()
        trainer.state.split_idx = 1
        trainer.state.called = "train"

        trainer._reset_validation(val_loader, metrics)

        assert trainer.state.split_idx == 1
        assert trainer.state.num_val_batches == 9
        assert trainer.state.called == "train"
        trainer.model.eval.assert_called_once()
        metrics.reset.assert_called_once_with(reset_df=False)

    def test_reset_validate(self, trainer: Trainer):
        val_loader = MagicMock()
        val_loader.__len__.return_value = 6
        metrics = Mock()

        trainer._reset_validate(1, val_loader, metrics)

        assert trainer.state.split_idx == 1
        assert trainer.state.num_val_batches == 6
        assert trainer.state.called == "validate"
        trainer.model.eval.assert_called_once()
        val_loader.dataset.eval.assert_called_once()
        metrics.reset.assert_called_once_with(reset_df=True)

    def test_reset_test(self, trainer: Trainer):
        test_loader = MagicMock()
        test_loader.__len__.return_value = 4
        metrics = Mock()

        trainer._reset_test(test_loader, metrics)

        assert trainer.state.num_test_batches == 4
        assert trainer.state.called == "test"
        trainer.model.eval.assert_called_once()
        test_loader.dataset.eval.assert_called_once()
        metrics.reset.assert_called_once_with(reset_df=True)

    def test_seed_context(self, trainer: Trainer):
        with trainer._seed_context(seed=None, deterministic=False):
            assert not os.environ.get("CLINICADL_GLOBAL_SEED")

        with trainer._seed_context(seed=1, deterministic=True):
            assert os.environ.get("CLINICADL_GLOBAL_SEED") == "1"
            assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"

        assert not os.environ.get("CLINICADL_GLOBAL_SEED")
        assert not os.environ.get("CLINICADL_DETERMINISTIC")

    def test_exception_context(self, trainer: Trainer):
        with pytest.raises(ValueError), trainer._exception_context():
            raise (e := ValueError())
        trainer.callbacks.callbacks[-3].on_exception.assert_called_once_with(
            model=ANY, maps=ANY, state=ANY, exception=e
        )

    def test_model_to(self, trainer: Trainer):
        comp_config = Mock()
        comp_config.device = torch.device("cpu")
        comp_config.non_blocking = True
        comp_config.channels_last = False

        trainer._model_to(comp_config)

        comp_config.check_device.assert_called()
        trainer.model.to.assert_called_once_with(
            device=torch.device("cpu"), non_blocking=True
        )

        trainer._model = torch.nn.Conv3d(2, 1, 1)
        assert next(iter(trainer._model.parameters())).stride() == (2, 1, 1, 1, 1)
        comp_config.channels_last = True
        trainer._model_to(comp_config)
        assert next(iter(trainer._model.parameters())).stride() == (2, 1, 2, 2, 2)

    def test_batch_to(self, trainer: Trainer):
        comp_config = Mock()
        comp_config.device = torch.device("cpu")
        comp_config.non_blocking = True
        comp_config.channels_last = True

        batch = MagicMock()
        batch.__class__ = Batch
        trainer._batch_to(batch, comp_config)
        batch.to.assert_called_once_with(
            device=torch.device("cpu"), non_blocking=True, channels_last=True
        )

        batch.reset_mock()
        trainer._batch_to((batch, batch), comp_config)
        batch.to.assert_has_calls(
            [
                call(device=torch.device("cpu"), non_blocking=True, channels_last=True),
                call(device=torch.device("cpu"), non_blocking=True, channels_last=True),
            ]
        )

        batch.reset_mock()
        trainer._batch_to({"b1": batch, "b2": batch}, comp_config)
        batch.to.assert_has_calls(
            [
                call(device=torch.device("cpu"), non_blocking=True, channels_last=True),
                call(device=torch.device("cpu"), non_blocking=True, channels_last=True),
            ]
        )

    def test_load_model_checkpoint(self, trainer: Trainer, tmp_path):
        intial_model = nn.Linear(1, 1)
        torch.save(intial_model.state_dict(), tmp_path / "model.pt")

        trainer._load_model_checkpoint(tmp_path / "model.pt")
        trainer.model.to.assert_called_once_with("cpu")

        trainer._model = nn.Linear(1, 1)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                next(iter(intial_model.parameters())),
                next(iter(trainer._model.parameters())),
            )
        trainer._load_model_checkpoint(tmp_path / "model.pt")
        torch.testing.assert_close(
            next(iter(intial_model.parameters())),
            next(iter(trainer._model.parameters())),
        )

    def test_call_event(self, trainer: Trainer):
        e = ValueError()
        trainer._call_event(
            "on_exception",
            exception=e,
        )
        trainer.callbacks.callbacks[-3].on_exception.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            exception=e,
        )

    def test_get_old_reprod_config(self, trainer: Trainer, tmp_path):
        maps = _add_maps_to_trainer(trainer, tmp_path)

        ComputationalConfig(seed=7, deterministic=True).to_json(
            maps.training.splits[0].computational_json, overwrite=True
        )
        seed, deterministic = trainer._get_old_reprod_config(split_idx=0)
        assert seed == 7
        assert deterministic

    def test_get_models_in_split(self, trainer: Trainer):
        maps = Maps(MAPS_PATH)
        maps.read()
        trainer._maps = maps
        iter_ = iter(trainer._get_models_in_split(0, model_checkpoint=None))
        assert next(iter_) == (
            "best-loss",
            maps.training.splits[0].models.best_models.metrics["loss"].model_pt,
        )
        assert next(iter_) == (
            "epoch-3",
            maps.training.splits[0].models.checkpoints.epochs[3].model_pt,
        )
        assert next(iter_) == ("final", maps.training.splits[0].models.final.model_pt)
        with pytest.raises(StopIteration):
            next(iter_)

        iter_ = iter(trainer._get_models_in_split(0, model_checkpoint="epoch-3"))
        assert next(iter_) == (
            "epoch-3",
            maps.training.splits[0].models.checkpoints.epochs[3].model_pt,
        )
        with pytest.raises(StopIteration):
            next(iter_)

    def test_get_dataloader(self, trainer: Trainer, tmp_path):
        maps = _add_maps_to_trainer(trainer, tmp_path)

        caps = CapsDataset(
            directory=CAPS_PATH,
            datatype=PETLinear(
                tracer="18FAV45",
                suvr_reference_region="pons2",
                use_uncropped_image=True,
            ),
            data=pd.DataFrame(
                {"participant_id": ["sub-000"], "session_id": ["ses-M000"]}
            ),
        )
        caps.read_tensor_conversion()
        caps.to_json(maps.training.data.train.splits[0].dataset_json, overwrite=True)
        DataLoaderConfig(batch_size=2).to_json(
            maps.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        dataloader = trainer._get_dataloader(maps.training.data.train.splits[0])
        assert isinstance(dataloader.dataset, CapsDataset)
        assert dataloader.batch_size == 2

        class CustomCollate(CollateFn):
            def __call__(self, samples):
                return super().__call__(samples)

        DataLoaderConfig(collate_fn=CustomCollate()).to_json(
            maps.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        with pytest.raises(
            CannotReadJsonError,
            match=f"ClinicaDL could not read the dataloader in {maps.training.data.train.splits[0].dataloader_json}. "
            "Please pass directly the dataloader via 'dataloader'.",
        ):
            trainer._get_dataloader(maps.training.data.train.splits[0])

        caps = CapsDataset(
            directory=CAPS_PATH,
            datatype=PETLinear(
                tracer="18FAV45",
                suvr_reference_region="pons2",
                use_uncropped_image=True,
            ),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000"],
                    "session_id": ["ses-M000"],
                    "age": [0.0],
                }
            ),
            columns={"age": lambda x: int(x)},
        )
        caps.to_json(maps.training.data.train.splits[0].dataset_json, overwrite=True)
        with pytest.raises(
            CannotReadJsonError,
            match=f"ClinicaDL could not read the dataset in {maps.training.data.train.splits[0].dataset_json}. "
            "Please pass directly the dataloader via 'dataloader'.",
        ):
            trainer._get_dataloader(maps.training.data.train.splits[0])

    def test_create_split(self, trainer: Trainer, tmp_path):
        maps = _add_maps_to_trainer(trainer, tmp_path)

        with pytest.raises(
            ValueError, match="Training on split 0 has already been performed."
        ):
            trainer._create_split(split_idx=0, resume=False)
        trainer._create_split(split_idx=0, resume=True)
        with pytest.raises(KeyError, match="Cannot resume training on split 2"):
            trainer._create_split(split_idx=2, resume=True)
        trainer._create_split(split_idx=2, resume=False)
        assert maps.training.splits[2].path.exists()

    def test_check_split_exists(self, trainer: Trainer):
        trainer._maps = Maps(MAPS_PATH)
        trainer._maps.read()
        with pytest.raises(KeyError, match="No training performed on split 2."):
            trainer._check_split_exists(2)
        trainer._check_split_exists(0)

    def test_check_group_exists(self, trainer: Trainer):
        trainer._maps = Maps(MAPS_PATH)
        trainer._maps.read()
        with pytest.raises(
            KeyError, match=re.escape("The group you passed ('abc') does not exist yet")
        ):
            trainer._check_group_exists("abc")
        trainer._check_group_exists("X")


class TestTrain:
    def test_train(self, trainer: Trainer, tmp_path):
        _add_maps_to_trainer(trainer, tmp_path)

        def _simulate_train(*args, **kwargs):
            assert torch.initial_seed() == 7
            assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"
            raise ValueError()

        trainer._train = Mock()
        split = Mock()
        split.index = 2

        trainer._train.side_effect = _simulate_train
        with pytest.raises(ValueError):
            trainer.train(
                split,
                computational=(comp := ComputationalConfig(seed=7, deterministic=True)),
                metrics=["loss"],
            )
        trainer._train.assert_called_once_with(
            split=split, computational=comp, metrics=["loss"], resume=False
        )
        trainer.callbacks.callbacks[-3].on_exception.assert_called()
        assert trainer.maps.training.splits[2].path.exists()
        assert torch.initial_seed() != 7
        assert not os.environ.get("CLINICADL_DETERMINISTIC")

        # resume
        trainer.maps.training.delete_split(2)
        split.index = 0
        trainer._get_old_reprod_config = Mock()
        trainer._get_old_reprod_config.side_effect = lambda x: (3, False)
        trainer._seed_context = Mock()
        trainer._seed_context.return_value = nullcontext()
        trainer._train = Mock()

        trainer.train(split, resume=True)

        trainer._seed_context.assert_called_once_with(3, False)

    def test__train(self, trainer: Trainer, custom_metric):
        trainer._metrics = MetricsHandler(loss=custom_metric, my_metric=custom_metric)
        trainer._metrics.init_metrics()
        trainer._model.build_optimizers.return_value = (optimizers := {"optim": Mock()})
        split = Mock()
        split.index = 0
        comp = Mock()
        comp.get_scaler.return_value = (scaler := Mock())
        trainer._reset_train = Mock()
        trainer._model_to = Mock()
        trainer._train_loop = Mock()

        trainer._train(split, computational=comp, metrics=["loss"], resume=False)

        metrics = trainer._train_loop.call_args.kwargs["metrics"]
        assert list(metrics.metrics.keys()) == ["loss"]

        trainer._reset_train.assert_called_once_with(split=split, metrics=metrics)
        trainer._model_to.assert_called_once_with(comp)
        trainer.callbacks.callbacks[-3].on_train_start.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            split=split,
            optimizers=optimizers,
            optimization=trainer.optimization,
            metrics=metrics,
            callbacks=trainer.callbacks,
            computational=comp,
        )
        trainer._train_loop.assert_called_once_with(
            split=split,
            optimizers=optimizers,
            grad_scaler=scaler,
            metrics=metrics,
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_train_end.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
        )

        # resume and metrics=None
        trainer._train_loop.reset_mock()
        trainer.callbacks.callbacks[-3].reset_mock()

        trainer._train(split, computational=comp, metrics=None, resume=True)

        trainer.callbacks.callbacks[-3].on_resume.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            split=split,
            optimizers=optimizers,
            grad_scaler=scaler,
            optimization=trainer.optimization,
            metrics=trainer.metrics,
            callbacks=trainer.callbacks,
            computational=comp,
        )
        trainer._train_loop.assert_called_once_with(
            split=split,
            optimizers=optimizers,
            grad_scaler=scaler,
            metrics=trainer.metrics,
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_train_end.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
        )

    @patch(
        "clinicadl.train.trainer.autocast",
        side_effect=lambda *args, **kwargs: nullcontext(),
    )
    def test_train_loop(self, autocast, trainer: Trainer):
        split = Mock()
        split.train_loader = MagicMock()
        split.train_loader.__len__.return_value = 3
        split.train_loader.__iter__.return_value = (batches := ["a", "b", "c"])
        split.val_loader = Mock()
        optimizers = {"opt1": Mock(), "opt2": Mock()}
        grad_scaler = Mock()
        metrics = Mock()
        comp = ComputationalConfig(amp=True)

        def early_stop(*, state, **kwargs):
            state.should_stop = state.current_epoch == 4

        trainer.callbacks.callbacks[-3].on_epoch_end.side_effect = early_stop
        trainer.state.current_epoch = 0
        trainer.optimization.num_epochs = 5
        trainer.optimization.accumulation_steps = 2
        trainer.optimization.evaluation_steps = 3

        trainer.model.forward_step.return_value = "loss"
        reset_epoch = trainer._reset_epoch
        trainer._reset_epoch = Mock()
        trainer._reset_epoch.side_effect = reset_epoch
        trainer._batch_to = Mock()
        trainer._validation = Mock()

        trainer._train_loop(split, optimizers, grad_scaler, metrics, comp)

        assert trainer.state.should_stop
        assert trainer.state.current_epoch == 4
        assert trainer.state.current_train_batch == 3
        assert trainer.state.optim_step == 4
        # epoch level
        trainer._reset_epoch.assert_has_calls(
            [
                call(1, train_loader=split.train_loader),
                call(2, train_loader=split.train_loader),
                call(3, train_loader=split.train_loader),
                call(4, train_loader=split.train_loader),
            ]
        )
        trainer.callbacks.callbacks[-3].on_epoch_start.assert_has_calls(
            [call(model=trainer.model, maps=trainer.maps, state=trainer.state)] * 4
        )
        trainer.callbacks.callbacks[-3].on_epoch_end.assert_has_calls(
            [call(model=trainer.model, maps=trainer.maps, state=trainer.state)] * 4
        )
        trainer._validation.assert_called_once_with(
            split.val_loader, metrics=metrics, computational=comp
        )
        # batch level
        trainer.callbacks.callbacks[-3].on_batch_start.assert_has_calls(
            [
                call(
                    model=trainer.model, maps=trainer.maps, state=trainer.state, batch=x
                )
                for x in batches * 4
            ]
        )
        trainer._batch_to.assert_has_calls(
            [
                call(
                    x,
                    computational=comp,
                )
                for x in batches * 4
            ]
        )
        trainer.callbacks.callbacks[-3].on_batch_end.assert_has_calls(
            [call(model=trainer.model, maps=trainer.maps, state=trainer.state)]
            * len(batches)
            * 4
        )
        # forward
        trainer.callbacks.callbacks[-3].on_forward_step_start.assert_has_calls(
            [
                call(
                    model=trainer.model, maps=trainer.maps, state=trainer.state, batch=x
                )
                for x in batches * 4
            ]
        )
        assert autocast.call_count == 12
        autocast.assert_called_with(device_type="cuda", enabled=True)
        trainer.model.forward_step.assert_has_calls(map(lambda x: call(x), batches * 4))
        # backward
        trainer.callbacks.callbacks[-3].on_backward_step_start.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    loss="loss",
                    grad_scaler=grad_scaler,
                )
            ]
            * len(batches)
            * 4
        )
        trainer.model.backward_step.assert_has_calls(
            [call("loss", grad_scaler=grad_scaler)] * len(batches) * 4
        )
        trainer.callbacks.callbacks[-3].on_backward_step_end.assert_has_calls(
            [call(model=trainer.model, maps=trainer.maps, state=trainer.state)]
            * len(batches)
            * 4
        )
        # optimization
        trainer.callbacks.callbacks[-3].on_optimization_step_start.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    optimizers=optimizers,
                    grad_scaler=grad_scaler,
                )
            ]
            * 4
        )
        trainer.model.optimization_step.assert_has_calls(
            [call(optimizers=optimizers, grad_scaler=grad_scaler)] * 4
        )
        assert trainer.model.optimization_step.call_count == 4
        assert optimizers["opt1"].zero_grad.call_count == 4
        assert optimizers["opt2"].zero_grad.call_count == 4
        trainer.callbacks.callbacks[-3].on_optimization_step_end.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    optimizers=optimizers,
                    grad_scaler=grad_scaler,
                )
            ]
            * 4
        )

    @patch(
        "clinicadl.train.trainer.autocast",
        side_effect=lambda *args, **kwargs: nullcontext(),
    )
    def test_train_loop_resume(self, autocast, trainer: Trainer):
        split = Mock()
        split.train_loader = MagicMock()
        split.train_loader.__len__.return_value = 3
        split.train_loader.__iter__.return_value = ["a", "b", "c"]
        optimizers = {"optimizer": Mock()}
        grad_scaler = Mock()
        metrics = Mock()
        comp = Mock()

        trainer.optimization.accumulation_steps = 1
        trainer.optimization.evaluation_steps = 1

        reset_epoch = trainer._reset_epoch
        trainer._reset_epoch = Mock()
        trainer._reset_epoch.side_effect = reset_epoch
        trainer._batch_to = Mock()
        trainer._validation = Mock()

        trainer._train_loop(split, optimizers, grad_scaler, metrics, comp)

        assert not trainer.state.should_stop
        assert trainer.state.current_epoch == 5
        assert trainer.state.current_train_batch == 3
        assert trainer.state.optim_step == 5 * 3
        trainer._validation.call_count == 5

    def test_train_loop_real(self, trainer: Trainer):
        computational = ComputationalConfig(amp=False, channels_last=False, gpu=False)
        self._real_test(trainer, computational)

    @pytest.mark.gpu
    def test_train_loop_real_gpu(self, trainer: Trainer):
        computational = ComputationalConfig(amp=True, channels_last=True, gpu=True)
        self._real_test(trainer, computational)

    def _real_test(self, trainer: Trainer, computational: ComputationalConfig):
        from clinicadl.optim import OptimizationConfig

        loader = _setup_dataloader()
        split = Mock()
        split.index = 0
        split.train_loader = loader

        trainer._validation = Mock()
        trainer._optim_config = OptimizationConfig(num_epochs=1)
        trainer._model = _setup_real_model()
        trainer._reset_train(split, metrics=Mock())
        trainer._model_to(computational)
        trainer.state.current_epoch = 0

        optimizers = trainer._model.build_optimizers()
        scaler = computational.get_scaler()

        trainer._train_loop(
            split, optimizers, scaler, metrics=Mock(), computational=computational
        )


class TestValidation:
    def test_validate(self, trainer: Trainer):
        trainer._maps = Mock()

        def _simulate_validate(*args, **kwargs):
            assert torch.initial_seed() == 1
            assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"
            raise ValueError()

        trainer._validate = Mock()
        trainer._check_split_exists = Mock()
        trainer._get_old_reprod_config = Mock()
        trainer._get_old_reprod_config.return_value = (1, True)

        trainer._validate.side_effect = _simulate_validate
        with pytest.raises(ValueError):
            trainer.validate(
                split_idx=0,
                metrics=["loss"],
                dataloader=(loader := Mock()),
                model_checkpoint="x",
                computational=(comp := Mock()),
            )
        trainer.maps.read.assert_called_once()
        trainer._check_split_exists.assert_called_once_with(0)
        trainer._validate.assert_called_once_with(
            split_idx=0,
            metrics=["loss"],
            dataloader=loader,
            model_checkpoint="x",
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_exception.assert_called()
        assert torch.initial_seed() != 7
        assert not os.environ.get("CLINICADL_DETERMINISTIC")

    def test__validate(self, custom_metric, trainer: Trainer):
        trainer._maps = Maps(MAPS_PATH)
        trainer.maps.read()
        trainer._metrics = MetricsHandler(loss=custom_metric, my_metric=custom_metric)
        trainer.metrics.init_metrics()

        trainer._evaluation_loop = Mock()
        trainer._check_split_exists = Mock()
        trainer._get_dataloader = Mock()
        trainer._get_dataloader.return_value = (loader := Mock())
        trainer._reset_validate = Mock()
        trainer._load_model_checkpoint = Mock()
        trainer._model_to = Mock()

        trainer._validate(
            split_idx=0,
            metrics=["loss"],
            dataloader=None,
            model_checkpoint="best-loss",
            computational=(comp := Mock()),
        )
        metrics = trainer._evaluation_loop.call_args.kwargs["metrics"]
        assert list(metrics.metrics.keys()) == ["loss"]

        trainer._get_dataloader.assert_called_once_with(
            trainer.maps.training.data.validation.splits[0]
        )
        trainer._reset_validate.assert_called_once_with(0, loader, metrics)
        trainer._load_model_checkpoint.assert_called_once_with(
            trainer.maps.training.splits[0].models.best_models.metrics["loss"].model_pt
        )
        trainer._model_to.assert_called_once_with(comp)
        trainer._evaluation_loop.assert_called_once_with(
            loader, metrics=metrics, computational=comp
        )
        trainer.callbacks.callbacks[-3].on_validate_start.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            dataloader=loader,
            model_checkpoint="best-loss",
            metrics=metrics,
            callbacks=trainer.callbacks,
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_validate_end.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            metrics=metrics,
        )

        # multiple checkpoints and dataloder
        trainer._get_dataloader.reset_mock()
        trainer._load_model_checkpoint.reset_mock()

        trainer._validate(
            split_idx=0,
            metrics=["loss"],
            dataloader=(loader := Mock()),
            model_checkpoint=None,
        )

        trainer._get_dataloader.assert_not_called()
        trainer._load_model_checkpoint.assert_has_calls(
            [
                call(
                    trainer.maps.training.splits[0]
                    .models.best_models.metrics["loss"]
                    .model_pt
                ),
                call(
                    trainer.maps.training.splits[0]
                    .models.checkpoints.epochs[3]
                    .model_pt
                ),
                call(trainer.maps.training.splits[0].models.final.model_pt),
            ]
        )
        trainer._evaluation_loop.assert_has_calls(
            [call(loader, metrics=ANY, computational=ANY)] * 3
        )

    def test_validation(self, trainer: Trainer):
        trainer._reset_validation = Mock()
        trainer._evaluation_loop = Mock()

        trainer.state.current_epoch = 2
        trainer._validation(
            (loader := Mock()),
            metrics=(metrics := Mock()),
            computational=(comp := Mock()),
        )

        trainer._reset_validation.assert_called_once_with(loader, metrics)
        trainer._evaluation_loop.assert_called_once_with(
            loader, metrics=metrics, epoch=2, computational=comp
        )
        trainer.callbacks.callbacks[-3].on_validation_start.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            dataloader=loader,
            metrics=metrics,
        )
        trainer.callbacks.callbacks[-3].on_validation_end.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            metrics=metrics,
        )


class TestTest:
    def test_test(self, trainer: Trainer):
        def _simulate_test(*args, **kwargs):
            assert torch.initial_seed() == 1
            assert os.environ.get("CLINICADL_DETERMINISTIC") == "true"
            raise ValueError()

        trainer._test = Mock()
        trainer._test.side_effect = _simulate_test
        with pytest.raises(ValueError):
            trainer.test(
                model_checkpoint="x",
                metrics=["loss"],
                group_name="y",
                dataloader=(loader := Mock()),
                computational=(comp := ComputationalConfig(deterministic=True, seed=1)),
            )
        trainer._test.assert_called_once_with(
            model_checkpoint="x",
            metrics=["loss"],
            group_name="y",
            dataloader=loader,
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_exception.assert_called()
        assert torch.initial_seed() != 7
        assert not os.environ.get("CLINICADL_DETERMINISTIC")

    def test__test(self, custom_metric, trainer: Trainer):
        trainer._maps = Maps(MAPS_PATH)
        trainer.maps.read()
        trainer._metrics = MetricsHandler(loss=custom_metric, my_metric=custom_metric)
        trainer.metrics.init_metrics()

        trainer._evaluation_loop = Mock()
        trainer._get_dataloader = Mock()
        trainer._get_dataloader.return_value = (loader := Mock())
        trainer._reset_test = Mock()
        trainer._load_model_checkpoint = Mock()
        trainer._model_to = Mock()

        trainer._test(
            model_checkpoint="split-0_best-loss",
            metrics=["loss"],
            group_name="X",
            dataloader=None,
            computational=(comp := Mock()),
        )

        metrics = trainer._evaluation_loop.call_args.kwargs["metrics"]
        assert list(metrics.metrics.keys()) == ["loss"]

        trainer._get_dataloader.assert_called_once_with(trainer.maps.test.groups["X"])
        trainer._reset_test.assert_called_once_with(loader, metrics)
        trainer._load_model_checkpoint.assert_called_once_with(
            trainer.maps.training.splits[0].models.best_models.metrics["loss"].model_pt
        )
        trainer._model_to.assert_called_once_with(comp)
        trainer._evaluation_loop.assert_called_once_with(
            loader, metrics=metrics, computational=comp
        )
        trainer.callbacks.callbacks[-3].on_test_start.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            dataloader=loader,
            model_checkpoint="split-0_best-loss",
            metrics=metrics,
            group_name="X",
            callbacks=trainer.callbacks,
            computational=comp,
        )
        trainer.callbacks.callbacks[-3].on_test_end.assert_called_once_with(
            model=trainer.model,
            maps=trainer.maps,
            state=trainer.state,
            metrics=metrics,
        )

        # dataloder and metrics=None
        trainer._get_dataloader.reset_mock()
        trainer._evaluation_loop.reset_mock()

        trainer._test(
            model_checkpoint="split-0_best-loss",
            metrics=None,
            group_name="X",
            dataloader=(loader := Mock()),
        )

        trainer._get_dataloader.assert_not_called()
        trainer._evaluation_loop.assert_called_once_with(
            ANY, metrics=trainer.metrics, computational=ANY
        )


class TestEvaluationLoop:
    @patch(
        "clinicadl.train.trainer.autocast",
        side_effect=lambda *args, **kwargs: nullcontext(),
    )
    def test_evaluation_loop(self, autocast, trainer: Trainer):
        def _simulate_evaluation(*args, **kwargs):
            assert not torch.is_grad_enabled()
            return "out"

        loader = MagicMock()
        loader.__len__.return_value = 3
        loader.__iter__.return_value = (batches := ["a", "b", "c"])

        trainer._batch_to = Mock()
        trainer.model.evaluation_step.side_effect = _simulate_evaluation
        metrics = Mock()
        metrics.return_value = "metrics"

        trainer._evaluation_loop(
            dataloader=loader,
            metrics=metrics,
            computational=(comp := ComputationalConfig(amp=True)),
            epoch=None,
        )

        trainer.state.current_val_batch == 3
        trainer._batch_to.assert_has_calls(
            [call(batch, computational=comp) for batch in batches]
        )
        trainer.model.evaluation_step.assert_has_calls(
            [call(batch) for batch in batches]
        )
        assert autocast.call_count == 3
        autocast.assert_called_with(device_type="cuda", enabled=True)
        metrics.assert_has_calls([call("out", epoch=None)] * len(batches))
        metrics.aggregate.assert_called_once_with(epoch=None)
        trainer.callbacks.callbacks[-3].on_batch_start.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    batch=batch,
                )
                for batch in batches
            ]
        )
        trainer.callbacks.callbacks[-3].on_evaluation_step_start.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    batch=batch,
                )
                for batch in batches
            ]
        )
        trainer.callbacks.callbacks[-3].on_metrics_computation_start.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    output="out",
                    metrics=metrics,
                )
            ]
            * len(batches)
        )
        trainer.callbacks.callbacks[-3].on_metrics_computation_end.assert_has_calls(
            [
                call(
                    model=trainer.model,
                    maps=trainer.maps,
                    state=trainer.state,
                    detailed_metrics_df="metrics",
                )
            ]
            * len(batches)
        )
        trainer.callbacks.callbacks[-3].on_batch_end.assert_has_calls(
            [call(model=trainer.model, maps=trainer.maps, state=trainer.state)]
            * len(batches)
        )

        # with epoch
        metrics.reset_mock()
        trainer._evaluation_loop(
            dataloader=loader,
            metrics=metrics,
            computational=(comp := ComputationalConfig(amp=True)),
            epoch=None,
        )
        metrics.assert_has_calls([call("out", epoch=None)] * len(batches))
        metrics.aggregate.assert_called_once_with(epoch=None)

    def test_evaluation_loop_real(self, trainer: Trainer):
        computational = ComputationalConfig(amp=False, channels_last=False, gpu=False)
        self._real_test(trainer, computational)

    @pytest.mark.gpu
    def test_evaluation_loop_real_gpu(self, trainer: Trainer):
        computational = ComputationalConfig(amp=True, channels_last=True, gpu=True)
        self._real_test(trainer, computational)

    def _real_test(self, trainer: Trainer, computational: ComputationalConfig):
        loader = _setup_dataloader()

        trainer._model = _setup_real_model()
        trainer._model_to(computational)
        metrics = MetricsHandler(metric=MSEMetricConfig())
        metrics.init_metrics()

        trainer._evaluation_loop(loader, metrics, computational=computational)


def test_from_maps(tmp_path, custom_metric):
    from clinicadl.callbacks import LoggerCallback
    from clinicadl.losses.config import MSELossConfig
    from clinicadl.metrics.config import MSEMetricConfig
    from clinicadl.models import SupervisedModel
    from clinicadl.networks.config import MLPConfig
    from clinicadl.optim import OptimizationConfig
    from clinicadl.optim.optimizers.config import AdamConfig

    model_with_config = SupervisedModel(
        network=MLPConfig(num_inputs=1, num_outputs=1, hidden_dims=[1]),
        loss=MSELossConfig(),
        optimizer=AdamConfig(),
    )

    Trainer(
        maps=tmp_path,
        model=model_with_config,
        overwrite=True,
    )
    trainer = Trainer.from_maps(tmp_path)
    assert trainer.maps.path == tmp_path
    assert trainer.model.config.network.value.num_inputs == 1
    assert trainer.model.config.network.value.num_inputs == 1
    assert trainer.metrics.config.metric_names == ["loss"]
    assert trainer.optimization.num_epochs == 10
    assert len(trainer.callbacks.config.callbacks) == 0

    Trainer(
        maps=tmp_path,
        model=model_with_config,
        optimization=OptimizationConfig(num_epochs=7),
        metrics={"mse": MSEMetricConfig()},
        callbacks=[LoggerCallback(save_logs=False)],
        overwrite=True,
    )
    trainer = Trainer.from_maps(tmp_path)
    assert trainer.metrics.config.metric_names == ["mse"]
    assert trainer.optimization.num_epochs == 7
    assert not trainer.callbacks.config.callbacks[0].config.save_logs

    # kwargs
    Trainer(
        maps=tmp_path,
        model=(model := _setup_real_model()),
        metrics=(metrics := {"mse": custom_metric}),
        callbacks=[cb := Callback()],
        overwrite=True,
    )
    with pytest.raises(
        CannotReadJsonError,
        match=re.escape(
            f"Cannot read the model (in {Maps(tmp_path).model_json}). Please pass it to from_maps via a keyword argument (e.g. Trainer.from_maps(..., model=...))."
        ),
    ):
        Trainer.from_maps(tmp_path)
    with pytest.raises(
        CannotReadJsonError,
        match="Cannot read the metrics",
    ):
        Trainer.from_maps(tmp_path, model=Mock())
    with pytest.raises(
        CannotReadJsonError,
        match="Cannot read the callbacks",
    ):
        Trainer.from_maps(tmp_path, model=Mock(), metrics=Mock())
    trainer = Trainer.from_maps(tmp_path, model=model, metrics=metrics, callbacks=[cb])
    assert trainer.model is model
    assert trainer.metrics.config.metrics.values["mse"].value is metrics["mse"]
    assert trainer.callbacks.config.callbacks[0] is cb
