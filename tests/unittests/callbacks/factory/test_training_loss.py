from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from clinicadl.callbacks.factory.training_loss import _TrainingLoss
from clinicadl.dictionary.words import BATCH, EPOCH, LOSS

# -----------------------
# Fake training state
# -----------------------


class FakeLogs:
    def __init__(self, tmp_path):
        self.training_tsv = tmp_path / "train_loss.tsv"


class FakeSplit:
    def __init__(self, tmp_path):
        self.logs = FakeLogs(tmp_path)


class FakeMaps:
    def __init__(self, tmp_path):
        self.training = type("Training", (), {})()
        self.training.splits = [FakeSplit(tmp_path)]


class FakeState:
    def __init__(self, tmp_path):
        self.maps = FakeMaps(tmp_path)
        self.split = type("Split", (), {"index": 0})()
        self.epoch = 0
        self.batch = 0


# -----------------------
# Tests
# -----------------------


@pytest.fixture
def training_loss_callback():
    return _TrainingLoss()


def test_on_batch_end_records_loss(training_loss_callback):
    state = FakeState(tmp_path=Path("."))
    callback = training_loss_callback

    # Simulate batch updates
    state.epoch = 0
    state.batch = 0
    callback.on_batch_end(state, loss=0.5)
    state.batch = 1
    callback.on_batch_end(state, loss=0.8)
    state.epoch = 1
    state.batch = 0
    callback.on_batch_end(state, loss=0.3)

    df = callback.df
    assert df.at[(0, 0), LOSS] == 0.5
    assert df.at[(0, 1), LOSS] == 0.8
    assert df.at[(1, 0), LOSS] == 0.3


def test_on_train_end_saves_file(tmp_path, training_loss_callback):
    state = FakeState(tmp_path=tmp_path)
    callback = training_loss_callback

    # Record a loss
    state.epoch = 0
    state.batch = 0
    callback.on_batch_end(state, loss=1.23)

    # Save to TSV
    callback.on_train_end(state)

    # Check file exists
    training_tsv = tmp_path / "train_loss.tsv"
    assert training_tsv.exists()

    # Check file content
    df_loaded = pd.read_csv(training_tsv, sep="\t", index_col=[0, 1])
    assert df_loaded.at[(0, 0), LOSS] == 1.23
