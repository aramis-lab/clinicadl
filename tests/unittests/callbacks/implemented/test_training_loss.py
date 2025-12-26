import re
import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from clinicadl.callbacks.implemented import TrainingLossCallback
from clinicadl.io import Maps
from clinicadl.train import TrainerState

MODEL = Mock()
LOSS = Mock()
MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"


def test_on_train_start():
    training_loss = TrainingLossCallback()

    MODEL.get_loss_functions.return_value = {}
    with pytest.raises(
        AssertionError,
        match="get_loss_functions method of you clinicadl.models.Model should return a dictionary with at least one key.",
    ):
        training_loss.on_train_start(model=MODEL)
    MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
    training_loss.on_train_start(model=MODEL)
    assert training_loss.df.columns.to_list() == ["my_loss"]


def test_on_backward_step_start():
    training_loss = TrainingLossCallback()
    state = TrainerState(current_train_batch=2, current_epoch=3)

    MODEL.get_loss_functions.return_value = {"my_loss": LOSS, "other_loss": LOSS}
    training_loss.on_train_start(model=MODEL)
    with pytest.raises(
        ValueError,
        match="forward_step should return a Tensor, or a dict of Tensors. Got:.*",
    ):
        training_loss.on_backward_step_start(state=state, loss=1.1)
    with pytest.raises(
        AssertionError,
        match="forward_step should return a Tensor, or a dict of Tensors. Got:.*",
    ):
        training_loss.on_backward_step_start(state=state, loss={"my_loss": 1.1})
    with pytest.raises(
        ValueError,
        match=re.escape(
            "clinicadl.models.Model.forward_step returns a single loss, whereas clinicadl.models.Model.get_loss_functions returns 2 loss function(s) ['my_loss', 'other_loss']"
        ),
    ):
        training_loss.on_backward_step_start(state=state, loss=torch.tensor([1.1]))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "clinicadl.models.Model.forward_step returns loss(es) named ['my_loss'], whereas clinicadl.models.Model.get_loss_functions "
            "returns ['my_loss', 'other_loss'] loss function(s)"
        ),
    ):
        training_loss.on_backward_step_start(
            state=state, loss={"my_loss": torch.tensor([1.1])}
        )

    training_loss.on_backward_step_start(
        state=state,
        loss={"my_loss": torch.tensor([1.1]), "other_loss": torch.tensor([1.2])},
    )
    state = TrainerState(current_train_batch=3, current_epoch=4)
    training_loss.on_backward_step_start(
        state=state,
        loss={"my_loss": torch.tensor([2.1]), "other_loss": torch.tensor([2.2])},
    )
    pd.testing.assert_frame_equal(
        training_loss.df.reset_index(),
        pd.DataFrame(
            {
                "epoch": [3, 4],
                "batch": [2, 3],
                "my_loss": [1.1, 2.1],
                "other_loss": [1.2, 2.2],
            }
        ),
    )

    # only one metric
    MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
    training_loss.on_train_start(model=MODEL)
    training_loss.on_backward_step_start(state=state, loss=torch.tensor([1.1]))
    pd.testing.assert_frame_equal(
        training_loss.df.reset_index(),
        pd.DataFrame(
            {
                "epoch": [4],
                "batch": [3],
                "my_loss": [1.1],
            }
        ),
    )


def test_on_train_end(tmp_path):
    shutil.copytree(MAPS_PATH, tmp_path, dirs_exist_ok=True)
    state = TrainerState(split_idx=1)
    maps = Maps(tmp_path)
    maps.training.create_split(1)
    training_loss = TrainingLossCallback()

    expected_df = pd.DataFrame(
        {
            "epoch": [4],
            "batch": [3],
            "my_loss": [1.1],
        }
    )
    training_loss.df = expected_df
    training_loss.on_train_end(maps, state)
    df = maps.load_file(maps.training.splits[1].logs.training_loss)
    pd.testing.assert_frame_equal(expected_df, df)


def test_to_from_dict():
    assert isinstance(
        TrainingLossCallback.from_dict(TrainingLossCallback().to_dict()),
        TrainingLossCallback,
    )


def test_state_dict():
    training_loss = TrainingLossCallback()
    state_dict = training_loss.state_dict()
    new_training_loss = TrainingLossCallback()
    new_training_loss.load_state_dict(state_dict)

    training_loss.df = pd.DataFrame(
        {
            "epoch": [4],
            "batch": [3],
            "my_loss": [1.1],
        }
    )
    state_dict = training_loss.state_dict()

    new_training_loss = TrainingLossCallback()
    new_training_loss.load_state_dict(state_dict)
    pd.testing.assert_frame_equal(new_training_loss.df, training_loss.df)
