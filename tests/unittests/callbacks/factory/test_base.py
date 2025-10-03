from importlib.util import find_spec

import pytest

from clinicadl.callbacks.factory.base import Callback, Tracker

# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def fake_state():
    """Minimal fake TrainerState for testing callbacks."""

    class DummyState:
        def __init__(self):
            self.current_loss = 0.1234

    return DummyState()


# -----------------------
# Tests for Callback
# -----------------------


class TestCallback:
    def test_to_dict_returns_class_name(self):
        cb = Callback()
        assert cb.to_dict() == {"name": "Callback"}

    def test_subclass_override(self, capsys, fake_state):
        class PrintLossCallback(Callback):
            def on_batch_end(self, config, **kwargs):
                print(f"Loss: {config.current_loss:.4f}")

        cb = PrintLossCallback()
        cb.on_batch_end(fake_state)

        captured = capsys.readouterr()
        assert "Loss: 0.1234" in captured.out

    def test_all_hooks_do_not_raise(self, fake_state):
        """Ensure base methods exist and can be called without error."""
        cb = Callback()

        hooks = [
            cb.on_train_begin,
            cb.on_train_end,
            cb.on_epoch_begin,
            cb.on_epoch_end,
            cb.on_batch_begin,
            cb.on_batch_end,
            cb.on_backward_begin,
            cb.on_backward_end,
            cb.on_validation_begin,
            cb.on_validation_end,
        ]

        for hook in hooks:
            hook(fake_state)  # should not raise


# -----------------------
# Tests for Tracker
# -----------------------


class TestTracker:
    def test_is_available_false_for_unknown_package(self):
        tracker = Tracker()
        tracker.package = "definitely_not_installed_package_xyz"
        assert tracker.is_available() is False

    def test_is_available_true_for_builtin_package(self):
        tracker = Tracker()
        tracker.package = "math"  # stdlib module, should exist
        assert tracker.is_available() is True

    def test_to_dict_inherits_callback(self):
        tracker = Tracker()
        assert tracker.to_dict() == {"name": "Tracker"}
