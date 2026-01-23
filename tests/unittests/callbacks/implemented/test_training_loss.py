import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import torch

from clinicadl.callbacks.implemented import TrainingLossCallback
from clinicadl.io import Maps
from clinicadl.train import TrainerState

MODEL = Mock()
LOSS = Mock()
MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"


def test_on_train_start():
    training_loss = TrainingLossCallback()
    state = TrainerState(split_idx=0)
    maps = Maps(MAPS_PATH)
    maps.read()

    MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
    training_loss.on_train_start(model=MODEL, state=state, maps=maps)
    assert training_loss.df.columns.to_list() == ["my_loss"]


def test_on_backward_step_start():
    training_loss = TrainingLossCallback()
    state = TrainerState(current_train_batch=2, current_epoch=3)

    MODEL.get_loss_functions.return_value = {"my_loss": LOSS, "other_loss": LOSS}
    training_loss.on_train_start(model=MODEL)

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
    state = TrainerState(split_idx=2)
    maps = Maps(tmp_path)
    maps.training.create_split(state.split_idx)
    training_loss = TrainingLossCallback()

    expected_df = pd.DataFrame(
        {
            "epoch": [4],
            "batch": [3],
            "my_loss": [1.1],
        }
    )
    training_loss.df = expected_df
    training_loss.on_train_end(maps=maps, state=state)
    df = maps.open_file(maps.training.splits[state.split_idx].logs.training_loss_tsv)
    pd.testing.assert_frame_equal(expected_df, df)


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
