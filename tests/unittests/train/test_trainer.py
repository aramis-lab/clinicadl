import os
import random
from unittest.mock import ANY, MagicMock, Mock, call

import pytest
import torch
import torch.nn as nn

from clinicadl.callbacks import Callback
from clinicadl.data.dataloader import Batch
from clinicadl.metrics import MetricsHandler
from clinicadl.train import Trainer


@pytest.fixture(autouse=True)
def trainer(tmp_path) -> Trainer:
    optim = Mock()
    optim.num_epochs = 5
    cb = Callback()
    cb.on_exception = Mock()
    return Trainer(
        maps_path=tmp_path / "maps",
        model=Mock(),
        metrics={},
        optimization=optim,
        callbacks=[Callback(), cb],
    )


class TestMethods:
    def test_reset_train(self, trainer: Trainer):
        split = Mock()
        split.index = 1
        metrics = Mock()

        trainer._reset_train(split, metrics)

        assert trainer.state.split_idx == 1
        assert trainer.state.num_epochs == 5
        assert trainer.state.called == "train"
        trainer.model.reset.assert_called_once()
        split.train_loader.dataset.train.assert_called_once()
        split.val_loader.dataset.eval.assert_called_once()
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
