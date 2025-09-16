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

    def test_set_tracker_uses_emissions_tracker(self, tmp_path):
        fake_state = FakeState(tmp_path)
        cc_instance = cc_module.CodeCarbon.__new__(cc_module.CodeCarbon)

        mock_tracker = MagicMock()
        with patch(
            "clinicadl.callbacks.factory.code_carbon.EmissionsTracker",
            return_value=mock_tracker,
        ):
            with patch(
                "clinicadl.callbacks.factory.code_carbon.OfflineEmissionsTracker"
            ) as mock_offline:
                cc_instance.set_tracker(fake_state)

        # Should have used EmissionsTracker, not Offline
        assert cc_instance.tracker == mock_tracker
        mock_offline.assert_not_called()

    def test_set_tracker_falls_back_to_offline(self, tmp_path):
        fake_state = FakeState(tmp_path)
        cc_instance = cc_module.CodeCarbon.__new__(cc_module.CodeCarbon)

        mock_offline = MagicMock()
        with patch(
            "clinicadl.callbacks.factory.code_carbon.EmissionsTracker",
            side_effect=Exception,
        ):
            with patch(
                "clinicadl.callbacks.factory.code_carbon.OfflineEmissionsTracker",
                return_value=mock_offline,
            ):
                cc_instance.set_tracker(fake_state)

        # Should have used OfflineEmissionsTracker
        assert cc_instance.tracker == mock_offline

    def test_on_train_begin_and_end(self, tmp_path):
        fake_state = FakeState(tmp_path)
        cc_instance = cc_module.CodeCarbon.__new__(cc_module.CodeCarbon)

        mock_tracker = MagicMock()
        with patch(
            "clinicadl.callbacks.factory.code_carbon.EmissionsTracker",
            return_value=mock_tracker,
        ):
            cc_instance.on_train_begin(fake_state)
            mock_tracker.start.assert_called_once()

            cc_instance.on_train_end(fake_state)
            mock_tracker.stop.assert_called_once()
