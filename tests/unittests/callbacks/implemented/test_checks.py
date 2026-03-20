import re
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.callbacks.implemented import ChecksCallback
from clinicadl.callbacks.implemented.checks import (
    _compare_dataloaders,
    _compare_datasets,
)
from clinicadl.data.dataloader import (
    Batch,
    CollateFn,
    DataLoaderConfig,
    MergeBatchesCollate,
)
from clinicadl.data.datasets import CapsDataset, UnpairedDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.io import Maps
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import DataFrameError, DataLeakageError
from clinicadl.utils.json import write_json


class Split(Mock):
    @property
    def train_loader(self):
        ...

    @property
    def val_loader(self):
        ...


MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
CAPS_PATH = Path(__file__).parents[2] / "resources" / "caps_example"
MAPS = Maps(MAPS_PATH)
CAPS = CapsDataset(
    directory=CAPS_PATH,
    datatype=PETLinear(
        tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
    ),
    data=CAPS_PATH / "tsv" / "labels.tsv",
    columns=["age"],
)
MAPS.read()
MODEL = Mock()
LOSS = Mock()
MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
SPLIT = Split()
SPLIT.index = 0
SPLIT.train_dataset = CAPS.subset([("sub-000", "ses-M000")])
SPLIT.val_dataset = CAPS.subset([("sub-010", "ses-M003")])
SPLIT.config.train_loader_config = DataLoaderConfig()
SPLIT.config.val_loader_config = DataLoaderConfig()
VAL_DATALOADER = Mock()
VAL_DATALOADER.dataset = CAPS.subset([("sub-010", "ses-M003")])
DATALOADER = Mock()
DATALOADER.dataset = CAPS.subset([("sub-100", "ses-M000")])
GROUP = "X"
MODEL_CHECKPOINT = "split-0_final"
STATE = Mock()
STATE.split_idx = SPLIT.index


class CustomCollate(CollateFn):
    def __call__(self, samples):
        pass


def create_new_maps(path):
    shutil.copytree(MAPS_PATH, path, dirs_exist_ok=True)
    maps = Maps(path)
    maps.read()

    return maps


class TestCheckInputs:
    checker = ChecksCallback()
    SPLIT = deepcopy(SPLIT)

    def test_on_train_start(self):
        self._split_check(self.checker.on_train_start)

    def test_on_resume(self):
        self._split_check(self.checker.on_resume)

    def _split_check(self, method):
        with patch.object(
            Split, "train_loader", new_callable=PropertyMock
        ) as mock_train, patch.object(
            Split, "val_loader", new_callable=PropertyMock
        ) as mock_val:
            method(split=self.SPLIT, model=MODEL, maps=MAPS)

        mock_train.assert_called_once()
        mock_val.assert_called_once()


class TestCheckLosses:
    checker = ChecksCallback()
    MODEL = Mock()

    def test_on_train_start(self):
        self.MODEL.get_loss_functions.return_value = {}
        with pytest.raises(
            ValueError,
            match="clinicadl.models.Model.get_loss_functions method should return a dictionary with at least one key.",
        ):
            self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)
        self.MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)

    def test_resume(self):
        self.checker.on_resume(model=self.MODEL, maps=MAPS, split=SPLIT)
        assert self.checker._check_losses._checked

    def test_on_backward_step_start(self):
        self.MODEL.get_loss_functions.return_value = {
            "my_loss": LOSS,
            "other_loss": LOSS,
        }
        self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)

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
        self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)
        self.checker.on_backward_step_start(loss=torch.tensor([1]))


class TestCheckDataFrames:
    checker = ChecksCallback()
    SPLIT = deepcopy(SPLIT)
    MAPS = Mock()
    DATALOADER = Mock()
    BAD_DF = pd.DataFrame({"participant_id": ["sub-000"], "session": ["ses-M000"]})

    def test_on_train_start(self):
        self._test_train(self.checker.on_train_start)

    def test_on_resume(self):
        self._test_train(self.checker.on_resume)

    def _test_train(self, method):
        self.SPLIT.train_dataset._df = self.BAD_DF
        self.SPLIT.val_dataset._df = SPLIT.val_dataset.df
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            method(split=self.SPLIT, model=MODEL, maps=MAPS)
        self.SPLIT.train_dataset._df = SPLIT.train_dataset.df
        self.SPLIT.val_dataset._df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            method(split=self.SPLIT, model=MODEL, maps=MAPS)

    def test_on_validate_start(self):
        self.DATALOADER.dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_validate_start(
                dataloader=self.DATALOADER,
                maps=MAPS,
                group_name=GROUP,
                model_checkpoint=MODEL_CHECKPOINT,
            )

    def test_on_test_start(self, tmp_path):
        MAPS = create_new_maps(tmp_path)
        MAPS.test.groups[GROUP].results.create_split(2)

        self.DATALOADER.dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_test_start(
                dataloader=self.DATALOADER,
                maps=MAPS,
                group_name=GROUP,
                model_checkpoint=MODEL_CHECKPOINT,
            )

    def test_on_predict_start(self, tmp_path):
        MAPS = create_new_maps(tmp_path)
        MAPS.prediction.groups[GROUP].results.create_split(2)

        self.DATALOADER.dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_predict_start(
                dataloader=self.DATALOADER,
                maps=MAPS,
                group_name=GROUP,
                model_checkpoint=MODEL_CHECKPOINT,
            )


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
            "participant_id": ["sub-000", "sub-010"],
            "session_id": ["ses-M001", "ses-M000"],
        }
    )

    def test_on_train_start(self):
        self.SPLIT.train_dataset.df = self.DF1
        self.SPLIT.val_dataset.df = self.BAD_DF2
        with pytest.raises(
            DataLeakageError,
            match=re.escape(
                "Some participants are in the training and validation sets: ['sub-000']"
            ),
        ):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)

    def test_on_test_start(self, tmp_path):
        MAPS = create_new_maps(tmp_path)
        MAPS.test.groups[GROUP].results.create_split(2)

        self.DATALOADER.dataset.df = self.BAD_DF2
        with pytest.raises(
            DataLeakageError,
            match=re.escape(
                f"Some test participants are in the training/validation participants: ['sub-000', 'sub-010'] (see: {MAPS.training.data.data_tsv})"
            ),
        ):
            self.checker.on_test_start(
                dataloader=self.DATALOADER,
                maps=MAPS,
                group_name=GROUP,
                model_checkpoint=MODEL_CHECKPOINT,
            )


class TestDataConsistency:
    checker = ChecksCallback()
    BAD_DATASET = CapsDataset(
        directory=CAPS_PATH,
        datatype=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=CAPS_PATH / "tsv" / "labels.tsv",
        transforms=TransformsHandler(image_transforms=[tio.ZNormalization()]),
        columns=["age"],
    )
    BAD_DATALOADER = DataLoaderConfig(batch_size=2, collate_fn=CustomCollate())
    SPLIT = deepcopy(SPLIT)
    STATE = Mock()

    def test_on_train_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.SPLIT.index = 2

        self.BAD_DATASET.subset([("sub-000", "ses-M000")]).to_json(
            MAPS.training.data.train.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATASET.subset([("sub-010", "ses-M003")]).to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATALOADER.to_json(
            MAPS.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        self.SPLIT.train_dataset.to_json(
            MAPS.training.data.train.splits[1].dataset_json, overwrite=True
        )
        self.SPLIT.val_dataset.to_json(
            MAPS.training.data.validation.splits[1].dataset_json, overwrite=True
        )
        self.SPLIT.config.train_loader_config.to_json(
            MAPS.training.data.train.splits[1].dataloader_json, overwrite=True
        )

        with caplog.at_level("WARNING"):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the training dataset of split-0 (in {MAPS.training.data.train.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[1].message == (
            f"Could not read the arguments ['transforms'] of the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[2].message == (
            f"Could not read the arguments ['collate_fn'] of the training dataloader of split-0 (in {MAPS.training.data.train.splits[0].dataloader_json}), "
            "and thus could not compare with the dataloader of split-2. Beware that differences between dataloaders could lead to inconsistent results across splits."
        )
        assert caplog.records[3].message == (
            "The training dataloaders of split-0 and the one of split-2 are different: the two dataloaders have different batch sizes. Got 1 and 2\n"
            "This may lead to inconsistent results across splits."
        )
        print(caplog.records)
        assert len(caplog.records) == 4

        write_json(
            MAPS.training.data.train.splits[0].dataset_json, {"abc": 0}, overwrite=True
        )
        write_json(
            MAPS.training.data.validation.splits[0].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        write_json(
            MAPS.training.data.train.splits[0].dataloader_json,
            {"abc": 0},
            overwrite=True,
        )

        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)
        assert caplog.records[0].message == (
            f"Could not read the training dataset of split-0 (in {MAPS.training.data.train.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[1].message == (
            f"Could not read the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[2].message == (
            f"Could not read the training dataloader of split-0 (in {MAPS.training.data.train.splits[0].dataloader_json}), "
            "and thus could not compare with the dataloader of split-2. Beware that differences between dataloaders could lead to inconsistent results across splits."
        )
        assert len(caplog.records) == 3

    def test_on_resume(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.SPLIT.index = 0
        self.SPLIT.train_dataset = CAPS.subset([("sub-000", "ses-M003")])
        self.SPLIT.val_dataset = CAPS.subset([("sub-010", "ses-M012")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The training dataset passed does not contain the same (participant, session) pairs as in the original training dataset of split-0 (in {MAPS.training.data.train.splits[0].data_tsv}). "
                "Difference: [('sub-000', 'ses-M000'), ('sub-000', 'ses-M003')]",
            ),
        ):
            self.checker.on_resume(maps=MAPS, split=self.SPLIT)
        self.SPLIT.train_dataset = CAPS.subset([("sub-000", "ses-M000")])
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The validation dataset passed does not contain the same (participant, session) pairs as in the original validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].data_tsv}). "
                "Difference: [('sub-010', 'ses-M003'), ('sub-010', 'ses-M012')]",
            ),
        ):
            self.checker.on_resume(maps=MAPS, split=self.SPLIT)
        self.SPLIT.val_dataset = CAPS.subset([("sub-010", "ses-M003")])

        self.SPLIT.train_dataset.to_json(
            MAPS.training.data.train.splits[0].dataset_json, overwrite=True
        )
        self.SPLIT.val_dataset.to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        self.SPLIT.config.train_loader_config.to_json(
            MAPS.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_resume(maps=MAPS, split=self.SPLIT)
        assert len(caplog.records) == 0

        self.BAD_DATASET.subset([("sub-000", "ses-M000")]).to_json(
            MAPS.training.data.train.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATASET.subset([("sub-010", "ses-M003")]).to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATALOADER.to_json(
            MAPS.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_resume(maps=MAPS, split=self.SPLIT)
        assert len(caplog.records) == 4
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the training dataset of split-0 (in {MAPS.training.data.train.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed for resuming training. Beware that differences between datasets could lead to inconsistent results."
        )
        assert caplog.records[1].message == (
            f"Could not read the arguments ['transforms'] of the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed for resuming training. Beware that differences between datasets could lead to inconsistent results."
        )
        assert caplog.records[2].message == (
            f"Could not read the arguments ['collate_fn'] of the training dataloader of split-0 (in {MAPS.training.data.train.splits[0].dataloader_json}), "
            "and thus could not compare with the dataloader passed for resuming training. Beware that differences between dataloaders could lead to inconsistent results."
        )
        assert caplog.records[3].message == (
            "The training dataloaders of split-0 and the one passed for resuming training are different: the two dataloaders have different batch sizes. Got 1 and 2\n"
            "This may lead to inconsistent results."
        )

        write_json(
            MAPS.training.data.train.splits[0].dataset_json, {"abc": 0}, overwrite=True
        )
        write_json(
            MAPS.training.data.validation.splits[0].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        write_json(
            MAPS.training.data.train.splits[0].dataloader_json,
            {"abc": 0},
            overwrite=True,
        )

        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_resume(maps=MAPS, split=self.SPLIT)
        assert caplog.records[0].message == (
            f"Could not read the training dataset of split-0 (in {MAPS.training.data.train.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed for resuming training. Beware that differences between datasets could lead to inconsistent results."
        )
        assert caplog.records[1].message == (
            f"Could not read the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed for resuming training. Beware that differences between datasets could lead to inconsistent results."
        )
        assert caplog.records[2].message == (
            f"Could not read the training dataloader of split-0 (in {MAPS.training.data.train.splits[0].dataloader_json}), "
            "and thus could not compare with the dataloader passed for resuming training. Beware that differences between dataloaders could lead to inconsistent results."
        )
        assert len(caplog.records) == 3

    def test_on_validate_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.STATE.split_idx = 0
        dataloader = Mock()
        dataloader.dataset = CAPS.subset([("sub-000", "ses-M000")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The validation dataset passed does not contain the same (participant, session) pairs as in the original validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].data_tsv}). "
                "Difference: [('sub-000', 'ses-M000'), ('sub-010', 'ses-M003')]",
            ),
        ):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=dataloader
            )

        VAL_DATALOADER.dataset.to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=VAL_DATALOADER
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=VAL_DATALOADER
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.validate. Beware that differences between datasets could lead to inconsistent results in validation metrics."
        )
        assert len(caplog.records) == 1

        write_json(
            MAPS.training.data.validation.splits[0].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=VAL_DATALOADER
            )
        assert caplog.records[0].message == (
            f"Could not read the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.validate. Beware that differences between datasets could lead to inconsistent results in validation metrics."
        )
        assert len(caplog.records) == 1

    def test_on_test_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        dataloader = Mock()
        dataloader.dataset = CAPS.subset([("sub-999", "ses-M999")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The test dataset passed does not contain the same (participant, session) pairs as in the original test dataset of group-X (in {MAPS.test.groups[GROUP].data_tsv}). "
                "Difference: [('sub-100', 'ses-M000'), ('sub-999', 'ses-M999')]",
            ),
        ):
            self.checker.on_test_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=dataloader,
                model_checkpoint=MODEL_CHECKPOINT,
            )

        DATALOADER.dataset.to_json(MAPS.test.groups[GROUP].dataset_json, overwrite=True)
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(MAPS.test.groups[GROUP].dataset_json, overwrite=True)
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the test dataset of group-X (in {MAPS.test.groups[GROUP].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.test. Beware that differences between datasets could lead to inconsistent results in test metrics."
        )
        assert len(caplog.records) == 1

        write_json(
            MAPS.test.groups[GROUP].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the test dataset of group-X (in {MAPS.test.groups[GROUP].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.test. Beware that differences between datasets could lead to inconsistent results in test metrics."
        )
        assert len(caplog.records) == 1

    def test_on_predict_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        dataloader = Mock()
        dataloader.dataset = CAPS.subset([("sub-999", "ses-M999")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The prediction dataset passed does not contain the same (participant, session) pairs as in the original prediction dataset of group-X (in {MAPS.prediction.groups[GROUP].data_tsv}). "
                "Difference: [('sub-100', 'ses-M000'), ('sub-999', 'ses-M999')]",
            ),
        ):
            self.checker.on_predict_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=dataloader,
                model_checkpoint=MODEL_CHECKPOINT,
            )

        DATALOADER.dataset.to_json(
            MAPS.prediction.groups[GROUP].dataset_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(
            MAPS.prediction.groups[GROUP].dataset_json, overwrite=True
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the prediction dataset of group-X (in {MAPS.prediction.groups[GROUP].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.predict. Beware that differences between datasets could lead to inconsistent results in predictions."
        )
        assert len(caplog.records) == 1

        write_json(
            MAPS.prediction.groups[GROUP].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name=GROUP,
                maps=MAPS,
                dataloader=DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the prediction dataset of group-X (in {MAPS.prediction.groups[GROUP].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.predict. Beware that differences between datasets could lead to inconsistent results in predictions."
        )
        assert len(caplog.records) == 1


class TestCompareDatasets:
    DATASET = CapsDataset(
        directory=CAPS_PATH,
        datatype=T1Linear(use_uncropped_image=True),
        data=pd.DataFrame(
            {
                "participant_id": ["sub-000"],
                "session_id": ["ses-M000"],
                "age": [1.0],
                "diagnosis": ["CN"],
            }
        ),
        columns=["age"],
    )

    def test_type(self):
        class CustomCaps(CapsDataset):
            pass

        dataset = CustomCaps(
            directory=self.DATASET.config.directory,
            datatype=self.DATASET.config.datatype,
            data=self.DATASET.config.data,
            columns=self.DATASET.config.columns,
        )
        assert re.match(
            "the two datasets are not the same type. Got .*CustomCaps'> and .*CapsDataset'>",
            _compare_datasets(dataset, self.DATASET, except_fields=[]),
        )

    def test_directory(self, tmp_path):
        shutil.copytree(CAPS_PATH, tmp_path, dirs_exist_ok=True)
        dataset = CapsDataset(
            tmp_path,
            datatype=self.DATASET.config.datatype,
            data=self.DATASET.config.data,
            columns=self.DATASET.config.columns,
        )
        assert re.match(
            f"the two datasets don't come from the same directory: {tmp_path} and {CAPS_PATH}",
            _compare_datasets(dataset, self.DATASET, except_fields=[]),
        )
        assert (
            _compare_datasets(dataset, self.DATASET, except_fields=["directory"])
            is None
        )

    @pytest.mark.parametrize(
        "name,arg,error_msg",
        [
            (
                "datatype",
                PETLinear(
                    tracer="18FAV45",
                    suvr_reference_region="pons2",
                    use_uncropped_image=True,
                ),
                "the two datasets don't have the same datatypes, which differ in their pattern or key. Got .*pet_linear.* and .*t1_linear.*",
            ),
            (
                "transforms",
                TransformsHandler(extraction=Slice()),
                "the two datasets don't have the same transforms. Got TransformsHandler configuration for slice extraction.*TransformsHandler configuration for image extraction.*",
            ),
            (
                "masks",
                ["brain"],
                "the two datasets don't have the same masks. Got {'brain'} and set()",
            ),
            (
                "columns",
                ["diagnosis"],
                "the two datasets don't have the same columns or column processing. Got {'diagnosis'} and {'age'}",
            ),
        ],
    )
    def test_other_args(self, name, arg, error_msg):
        dataset = deepcopy(self.DATASET)
        setattr(dataset.config, name, arg)
        assert _compare_datasets(dataset, self.DATASET, except_fields=[name]) is None
        assert re.match(
            error_msg,
            _compare_datasets(dataset, self.DATASET, except_fields=[]),
            re.DOTALL,
        )

    def test_collection_dataset(self):
        self.DATASET.read_tensor_conversion()
        dataset = deepcopy(self.DATASET)
        dataset.config.masks = ["brain"]
        error_msg = _compare_datasets(
            UnpairedDataset([self.DATASET, self.DATASET]),
            UnpairedDataset([self.DATASET, self.DATASET]),
            except_fields=[],
        )
        assert error_msg is None
        error_msg = _compare_datasets(
            UnpairedDataset([dataset, dataset]),
            UnpairedDataset([self.DATASET, self.DATASET]),
            except_fields=[],
        )
        assert (
            error_msg
            == "the two datasets don't have the same masks. Got {'brain'} and set()"
        )
        error_msg = _compare_datasets(
            UnpairedDataset([dataset, dataset]),
            UnpairedDataset([self.DATASET, self.DATASET]),
            except_fields=["masks"],
        )
        assert error_msg is None


@pytest.mark.parametrize(
    "name,arg,error_msg",
    [
        (
            "batch_size",
            2,
            "the two dataloaders have different batch sizes. Got 2 and 1",
        ),
        (
            "sampling_weights",
            "abc",
            "the two dataloaders have different sampling weight columns. Got abc and None",
        ),
        (
            "shuffle",
            False,
            "one dataloader is shuffled but the other is not.",
        ),
        (
            "drop_last",
            True,
            "one dataloader has drop_last=True whereas the other has drop_last=False.",
        ),
        (
            "collate_fn",
            MergeBatchesCollate(),
            "the two dataloaders have different collate function. Got .*MergeBatchesCollate.* and None",
        ),
    ],
)
def test_compare_dataloaders(name, arg, error_msg):
    ref = DataLoaderConfig()
    dataloader = DataLoaderConfig(**{name: arg})
    assert _compare_dataloaders(dataloader, ref, except_fields=[name]) is None
    assert re.match(
        error_msg, _compare_dataloaders(dataloader, ref, except_fields=[]), re.DOTALL
    )


class TestBatch:
    checker = ChecksCallback()
    BATCH = Batch(["x"])

    def test_train(self):
        self._test_phase(
            self.checker.on_train_start, {"split": SPLIT, "model": MODEL, "maps": MAPS}
        )

    def test_resume(self):
        self._test_phase(self.checker.on_resume, {"split": SPLIT, "maps": MAPS})

    def test_validate(self):
        self._test_phase(
            self.checker.on_validate_start,
            {"state": STATE, "maps": MAPS, "dataloader": VAL_DATALOADER},
        )

    def test_test(self):
        self._test_phase(
            self.checker.on_test_start,
            {
                "dataloader": DATALOADER,
                "maps": MAPS,
                "group_name": GROUP,
                "model_checkpoint": MODEL_CHECKPOINT,
            },
        )

    def test_predict(self):
        self._test_phase(
            self.checker.on_predict_start,
            {
                "dataloader": DATALOADER,
                "maps": MAPS,
                "group_name": GROUP,
                "model_checkpoint": MODEL_CHECKPOINT,
            },
        )

    def _test_phase(self, method, args):
        method(**args)
        with pytest.raises(
            ValueError,
            match="The batch returned by your dataloader can be either a Batch, a sequence of Batch, "
            "or a dict or Batch. Got: abc",
        ):
            self.checker.on_batch_start(batch="abc")

        self.checker.on_batch_start(batch=self.BATCH)
        method(**args)
        self.checker.on_batch_start(batch=[self.BATCH])
        method(**args)
        self.checker.on_batch_start(batch={"x": self.BATCH})
        self.checker.on_batch_start(batch="abc")  # already checked
