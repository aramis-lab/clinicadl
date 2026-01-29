from unittest.mock import Mock

import pytest
import torch

from clinicadl.train import Trainer


class TestMethods:
    @pytest.fixture(autouse=True)
    def trainer(tmp_path) -> Trainer:
        return Trainer(
            maps_path=tmp_path,
            model=Mock(),
            metrics=Mock(),
            optimization=Mock(),
            callbacks=Mock(),
        )

    # def test_reset_train(self):
