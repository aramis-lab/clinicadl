import re
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from clinicadl.callbacks.implemented import ChecksCallback
from clinicadl.io import Maps
from clinicadl.utils.exceptions import DataFrameError, DataLeakageError

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
MAPS = Maps(MAPS_PATH)
MODEL = Mock()
LOSS = Mock()
MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
SPLIT = Mock()
DATALOADER = Mock()
SPLIT.train_dataset.df = pd.DataFrame(
    {
        "participant_id": ["sub-000", "sub-001"],
        "session_id": ["ses-M000", "ses-M006"],
    }
)
SPLIT.val_dataset.df = pd.DataFrame(
    {
        "participant_id": ["sub-002", "sub-002"],
        "session_id": ["ses-M001", "ses-M000"],
    }
)
DATALOADER.dataset.df = pd.DataFrame(
    {
        "participant_id": ["sub-003", "sub-004"],
        "session_id": ["ses-M000", "ses-M000"],
    }
)


class TestCheckLosses:
    checker = ChecksCallback()
    MODEL = Mock()

    def test_on_train_start(self):
        self.MODEL.get_loss_functions.return_value = {}
        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.get_loss_functions method should return a dictionary with at least one key.",
        ):
            self.checker.on_train_start(model=self.MODEL, split=SPLIT)
        self.MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        self.checker.on_train_start(model=self.MODEL, split=SPLIT)

    def test_on_backward_step_start(self):
        self.MODEL.get_loss_functions.return_value = {
            "my_loss": LOSS,
            "other_loss": LOSS,
        }
        self.checker.on_train_start(model=self.MODEL, split=SPLIT)

        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.forward_step should return a Tensor, or a dict of Tensors. Got:.*",
        ):
            self.checker.on_backward_step_start(loss=1.1)
        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.forward_step should return a Tensor, or a dict of Tensors. Got:.*",
        ):
            self.checker.on_backward_step_start(loss={"my_loss": 1.1})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "clinicadl.models.Model.forward_step returns a single loss, whereas clinicadl.models.Model.get_loss_functions returns 2 loss function(s) ['my_loss', 'other_loss']"
            ),
        ):
            self.checker.on_backward_step_start(loss=torch.tensor([1.1]))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "clinicadl.models.Model.forward_step returns loss(es) named ['my_loss'], whereas clinicadl.models.Model.get_loss_functions "
                "returns ['my_loss', 'other_loss'] loss function(s)"
            ),
        ):
            self.checker.on_backward_step_start(loss={"my_loss": torch.tensor([1.1])})

        self.checker.on_backward_step_start(
            loss={"my_loss": torch.tensor([1.1]), "other_loss": torch.tensor([1.2])},
        )
        assert self.checker._check_losses._checked
        self.checker.on_backward_step_start(
            loss={"my_loss": torch.tensor([1.1]), "other_loss": torch.tensor([1.2])},
        )
        assert self.checker._check_losses._checked

        self.MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        self.checker.on_train_start(model=self.MODEL, split=SPLIT)
        self.checker.on_backward_step_start(loss=torch.tensor([1]))


class TestCheckDataFrames:
    checker = ChecksCallback()
    SPLIT = Mock()
    MAPS = Mock()
    DATALOADER = Mock()
    BAD_DF = pd.DataFrame({"participant_id": ["sub-000"], "session": ["ses-M000"]})

    def test_on_train_start(self):
        self.SPLIT.train_dataset.df = self.BAD_DF
        self.SPLIT.val_dataset.df = SPLIT.val_dataset.df
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_train_start(split=self.SPLIT)
        self.SPLIT.train_dataset.df = SPLIT.train_dataset.df
        self.SPLIT.val_dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_train_start(split=self.SPLIT)

    def test_on_test_start(self):
        self.DATALOADER.dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_test_start(dataloader=self.DATALOADER)

    def test_on_predict_start(self):
        self.DATALOADER.dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_predict_start(dataloader=self.DATALOADER)


class TestDataLeakage:
    checker = ChecksCallback()
    SPLIT = Mock()
    DATALOADER = Mock()
    DF1 = pd.DataFrame(
        {
            "participant_id": ["sub-000", "sub-001"],
            "session_id": ["ses-M000", "ses-M006"],
        }
    )
    BAD_DF2 = pd.DataFrame(
        {
            "participant_id": ["sub-000", "sub-002"],
            "session_id": ["ses-M001", "ses-M000"],
        }
    )

    def test_on_train_start(self):
        self.SPLIT.train_dataset.df = self.DF1
        self.SPLIT.val_dataset.df = self.BAD_DF2
        with pytest.raises(
            DataLeakageError,
            match=f"Some participants are in the training and validation sets: {{'sub-000'}}",
        ):
            self.checker.on_train_start(split=self.SPLIT)

    def test_on_test_start(self):
        self.DATALOADER.dataset.df = self.BAD_DF2
        with pytest.raises(
            DataLeakageError,
            match=re.escape(
                f"Some test participants are in the training/validation participants: {{'sub-000'}} (see: {MAPS.training.data.data_tsv})"
            ),
        ):
            self.checker.on_test_start(dataloader=self.DATALOADER, maps=MAPS)


def test_state_dict():
    ChecksCallback().load_state_dict(ChecksCallback().state_dict())
