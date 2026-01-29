from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from clinicadl.callbacks.implemented import ConfigSaverCallback
from clinicadl.io import Maps

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"


def test_on_trainer_init(tmp_path):
    MODEL = Mock()
    METRICS = Mock()
    OPTIMIZATION = Mock()
    CALLBACKS = Mock()

    maps = Maps(tmp_path)

    saver = ConfigSaverCallback()
    saver.on_trainer_init(
        model=MODEL,
        metrics=METRICS,
        optimization=OPTIMIZATION,
        callbacks=CALLBACKS,
        maps=maps,
    )
    MODEL.to_json.assert_called_once_with(maps.model_json, overwrite=False)
    METRICS.to_json.assert_called_once_with(maps.metrics_json)
    OPTIMIZATION.to_json.assert_called_once_with(maps.training.optimization_json)
    CALLBACKS.to_json.assert_called_once_with(maps.callbacks_json)

    MODEL.reset_mock()
    METRICS.reset_mock()
    OPTIMIZATION.reset_mock()
    CALLBACKS.reset_mock()
    maps = Maps(MAPS_PATH)
    saver.on_trainer_init(
        model=MODEL,
        metrics=METRICS,
        optimization=OPTIMIZATION,
        callbacks=CALLBACKS,
        maps=maps,
    )
    MODEL.to_json.assert_not_called()
    METRICS.to_json.assert_not_called()
    OPTIMIZATION.to_json.assert_not_called()
    CALLBACKS.to_json.assert_not_called()


def test_on_train_start(tmp_path):
    SPLIT = Mock()
    SPLIT.index = 2
    SPLIT.train_dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-100", "sub-100", "sub-000"],
            "session_id": ["ses-M000", "ses-M000", "ses-M000"],
            "abc": "xxx",
        }
    )
    SPLIT.val_dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-101", "sub-101"],
            "session_id": ["ses-M000", "ses-M000"],
        }
    )
    COMPUTATIONAL = Mock()

    maps = Maps(tmp_path)
    maps.training.create_split(SPLIT.index)

    saver = ConfigSaverCallback()
    saver.on_train_start(maps=maps, split=SPLIT, computational=COMPUTATIONAL)

    SPLIT.train_dataset.to_json.assert_called_once_with(
        maps.training.data.train.splits[SPLIT.index].dataset_json, overwrite=False
    )
    SPLIT.config.train_loader_config.to_json.assert_called_once_with(
        maps.training.data.train.splits[SPLIT.index].dataloader_json
    )
    SPLIT.config.val_loader_config.to_json.assert_called_once_with(
        maps.training.data.validation.splits[SPLIT.index].dataloader_json
    )
    SPLIT.val_dataset.to_json.assert_called_once_with(
        maps.training.data.validation.splits[SPLIT.index].dataset_json, overwrite=False
    )
    COMPUTATIONAL.to_json.assert_called_once_with(
        maps.training.splits[SPLIT.index].computational_json
    )

    train_df = maps.open_file(maps.training.data.train.splits[SPLIT.index].data_tsv)
    val_df = maps.open_file(maps.training.data.validation.splits[SPLIT.index].data_tsv)
    global_df = maps.open_file(maps.training.data.data_tsv)
    pd.testing.assert_frame_equal(
        train_df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100"],
                "session_id": ["ses-M000", "ses-M000"],
            }
        ),
    )
    pd.testing.assert_frame_equal(
        val_df,
        pd.DataFrame({"participant_id": ["sub-101"], "session_id": ["ses-M000"]}),
    )
    pd.testing.assert_frame_equal(
        global_df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100", "sub-101"],
                "session_id": ["ses-M000", "ses-M000", "ses-M000"],
            }
        ),
    )

    SPLIT.index = 3
    SPLIT.val_dataset.df = pd.DataFrame(
        {"participant_id": ["sub-102"], "session_id": ["ses-M000"]}
    )
    maps.training.create_split(SPLIT.index)
    saver.on_train_start(maps=maps, split=SPLIT, computational=COMPUTATIONAL)
    global_df = maps.open_file(maps.training.data.data_tsv)
    print(global_df)
    pd.testing.assert_frame_equal(
        global_df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-100", "sub-101", "sub-102"],
                "session_id": ["ses-M000", "ses-M000", "ses-M000", "ses-M000"],
            }
        ),
    )


def test_on_test_start(tmp_path):
    DATALOADER = Mock()
    DATALOADER.dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-100", "sub-000"],
            "session_id": ["ses-100", "ses-M000"],
            "abc": "xxx",
        }
    )
    EXPECTED_DF = pd.DataFrame(
        {
            "participant_id": ["sub-000", "sub-100"],
            "session_id": ["ses-M000", "ses-100"],
        }
    )
    maps = Maps(tmp_path)

    saver = ConfigSaverCallback()
    saver.on_test_start(maps=maps, dataloader=DATALOADER, group_name="X")
    DATALOADER.dataset.to_json.assert_called_once_with(
        maps.test.groups["X"].dataset_json, overwrite=False
    )
    DATALOADER.config.to_json.assert_called_once_with(
        maps.test.groups["X"].dataloader_json
    )
    df = maps.open_file(maps.test.groups["X"].data_tsv)
    pd.testing.assert_frame_equal(
        df,
        EXPECTED_DF,
    )

    DATALOADER.dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-999"],
            "session_id": ["ses-999"],
        }
    )
    saver.on_test_start(maps=maps, dataloader=DATALOADER, group_name="X")
    df = maps.open_file(maps.test.groups["X"].data_tsv)
    pd.testing.assert_frame_equal(
        df,
        EXPECTED_DF,
    )


def test_on_predict_start(tmp_path):
    DATALOADER = Mock()
    DATALOADER.dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-100", "sub-000"],
            "session_id": ["ses-100", "ses-M000"],
            "abc": "xxx",
        }
    )
    EXPECTED_DF = pd.DataFrame(
        {
            "participant_id": ["sub-000", "sub-100"],
            "session_id": ["ses-M000", "ses-100"],
        }
    )
    maps = Maps(tmp_path)

    saver = ConfigSaverCallback()
    saver.on_predict_start(maps=maps, dataloader=DATALOADER, group_name="X")
    DATALOADER.dataset.to_json.assert_called_once_with(
        maps.prediction.groups["X"].dataset_json, overwrite=False
    )
    DATALOADER.config.to_json.assert_called_once_with(
        maps.prediction.groups["X"].dataloader_json
    )
    df = maps.open_file(maps.prediction.groups["X"].data_tsv)
    pd.testing.assert_frame_equal(
        df,
        EXPECTED_DF,
    )

    DATALOADER.dataset.df = pd.DataFrame(
        {
            "participant_id": ["sub-999"],
            "session_id": ["ses-999"],
        }
    )
    saver.on_predict_start(maps=maps, dataloader=DATALOADER, group_name="X")
    df = maps.open_file(maps.prediction.groups["X"].data_tsv)
    pd.testing.assert_frame_equal(
        df,
        EXPECTED_DF,
    )
