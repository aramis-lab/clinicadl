from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import clinicadl.callbacks.factory.code_carbon as cc_module


class FakeSplit:
    def __init__(self, path: Path):
        self.path = path
        self.index = 0


class FakeTraining:
    def __init__(self, path: Path):
        self.splits = [FakeSplit(path)]


class FakeMaps:
    def __init__(self, path: Path):
        self.training = FakeTraining(path)


class FakeState:
    def __init__(self, tmp_path: Path):
        self.maps = FakeMaps(tmp_path)
        self.split = FakeSplit(tmp_path)


class TestCodeCarbon:
    def test_is_available_true_when_codecarbon_exists(self):
        with patch(
            "clinicadl.callbacks.factory.code_carbon.find_spec", return_value=True
        ):
            assert cc_module.CodeCarbon.is_available() is True

    def test_is_available_false_when_codecarbon_missing(self):
        with patch(
            "clinicadl.callbacks.factory.code_carbon.find_spec", return_value=None
        ):
            assert cc_module.CodeCarbon.is_available() is False

    def test_init_raises_if_not_available(self):
        with patch.object(cc_module.CodeCarbon, "is_available", return_value=False):
            with pytest.raises(ModuleNotFoundError):
                cc_module.CodeCarbon()
