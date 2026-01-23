import re
import shutil
from copy import copy, deepcopy
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.callbacks.implemented import ChecksCallback
from clinicadl.callbacks.implemented.checks import (
    _compare_dataloaders,
    _compare_datasets,
)
from clinicadl.data.dataloader import CollateFn, DataLoaderConfig, MergeBatchesCollate
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.io import Maps
from clinicadl.transforms.extraction import Slice
from clinicadl.transforms.handlers import Transforms
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import DataFrameError, DataLeakageError
from clinicadl.utils.json import write_json
from clinicadl.utils.objects import HasConfig

MAPS_PATH = Path(__file__).parents[2] / "resources" / "maps_example"
CAPS_PATH = Path(__file__).parents[2] / "resources" / "caps_example"
MAPS = Maps(MAPS_PATH)
MAPS.read()
MODEL = Mock()
LOSS = Mock()
MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
DATALOADER = Mock()
SPLIT = Mock()
SPLIT.index = 2
SPLIT.train_dataset.__class__ = HasConfig
SPLIT.val_dataset.__class__ = HasConfig
SPLIT.config.train_loader_config = ClinicaDLConfig()
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


class TestInputs:
    checker = ChecksCallback()
    SPLIT = copy(SPLIT)

    def test_on_train_start(self):
        self.SPLIT.index = 1
        with pytest.raises(
            ValueError,
            match="Training on split 1. To relaunch a training on this split, first delete it properly with clinicadl.io.Maps.delete_split",
        ):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)

        self.SPLIT.index = 2
        self.SPLIT.train_loader = None
        self.SPLIT.val_loader = Mock()
        with pytest.raises(
            RuntimeError,
            match="The split has no training dataloder defined. Please run 'build_train_loader'",
        ):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)

        self.SPLIT.train_loader = Mock()
        self.SPLIT.val_loader = None
        with pytest.raises(
            RuntimeError,
            match="The split has no validation dataloder defined. Please run 'build_val_loader'",
        ):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)

    def test_on_test_start(self):
        with pytest.raises(
            FileExistsError,
            match=re.escape(
                f"There are already some results for checkpoint 'best-loss' in {MAPS.test.groups['X'].results.splits[0].path}. "
                f"Set overwrite=True in Trainer.test to overwrite them."
            ),
        ):
            self.checker.on_test_start(
                dataloader=DATALOADER,
                maps=MAPS,
                group_name="X",
                model_checkpoint="split-0_best-loss",
            )

    def test_on_predict_start(self):
        with pytest.raises(
            FileExistsError,
            match=re.escape(
                f"There are already some results for checkpoint 'best-loss' in {MAPS.prediction.groups['X'].results.splits[0].path}. "
                f"Set overwrite=True in Trainer.predict to overwrite them."
            ),
        ):
            self.checker.on_predict_start(
                dataloader=DATALOADER,
                maps=MAPS,
                group_name="X",
                model_checkpoint="split-0_best-loss",
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
            self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)
        self.MODEL.get_loss_functions.return_value = {"my_loss": LOSS}
        print(isinstance(SPLIT.train_dataset, HasConfig))
        print(isinstance(SPLIT.val_dataset, HasConfig))
        print(isinstance(SPLIT.config.train_loader_config, ClinicaDLConfig))
        self.checker.on_train_start(model=self.MODEL, split=SPLIT, maps=MAPS)

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
    SPLIT = copy(SPLIT)
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
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)
        self.SPLIT.train_dataset.df = SPLIT.train_dataset.df
        self.SPLIT.val_dataset.df = self.BAD_DF
        with pytest.raises(
            DataFrameError,
            match="The DataFrame of your clinicadl.data.dataset.Dataset is not valid.",
        ):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)

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
    GOOD_DATASET = CapsDataset(
        directory=CAPS_PATH,
        datatype=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=CAPS_PATH / "tsv" / "labels.tsv",
    )
    BAD_DATASET = CapsDataset(
        directory=CAPS_PATH,
        datatype=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=CAPS_PATH / "tsv" / "labels.tsv",
        transforms=Transforms(image_transforms=[tio.ZNormalization()]),
        label="age",
        columns=["age"],
    )
    GOOD_DATALOADER = DataLoaderConfig()
    BAD_DATALOADER = DataLoaderConfig(batch_size=2, collate_fn=CustomCollate())

    SPLIT = copy(SPLIT)
    STATE = Mock()
    DATALOADER = Mock()

    def test_on_train_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.SPLIT.train_dataset = self.GOOD_DATASET.subset([("sub-000", "ses-M000")])
        self.SPLIT.val_dataset = self.GOOD_DATASET.subset([("sub-010", "ses-M003")])
        self.SPLIT.config.train_loader_config = self.GOOD_DATALOADER

        self.BAD_DATASET.subset([("sub-000", "ses-M000")]).to_json(
            MAPS.training.data.train.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATASET.subset([("sub-010", "ses-M003")]).to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        self.BAD_DATALOADER.to_json(
            MAPS.training.data.train.splits[0].dataloader_json, overwrite=True
        )
        self.GOOD_DATASET.subset([("sub-000", "ses-M003")]).to_json(
            MAPS.training.data.train.splits[1].dataset_json, overwrite=True
        )
        self.GOOD_DATASET.subset([("sub-010", "ses-M012")]).to_json(
            MAPS.training.data.validation.splits[1].dataset_json, overwrite=True
        )
        self.GOOD_DATALOADER.to_json(
            MAPS.training.data.train.splits[1].dataloader_json, overwrite=True
        )

        with caplog.at_level("WARNING"):
            self.checker.on_train_start(split=self.SPLIT, model=MODEL, maps=MAPS)
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the training dataset of split-0 (in {MAPS.training.data.train.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[1].message == (
            "The training datasets of split-0 and the one of split-2 are different: the two datasets don't have the same label. Got None and age\n"
            "This may lead to inconsistent results across splits."
        )
        assert caplog.records[2].message == (
            f"Could not read the arguments ['transforms'] of the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset of split-2. Beware that differences between datasets could lead to inconsistent results across splits."
        )
        assert caplog.records[3].message == (
            "The validation datasets of split-0 and the one of split-2 are different: the two datasets don't have the same label. Got None and age\n"
            "This may lead to inconsistent results across splits."
        )
        assert caplog.records[4].message == (
            f"Could not read the arguments ['collate_fn'] of the training dataloader of split-0 (in {MAPS.training.data.train.splits[0].dataloader_json}), "
            "and thus could not compare with the dataloader of split-2. Beware that differences between dataloaders could lead to inconsistent results across splits."
        )
        assert caplog.records[5].message == (
            "The training dataloaders of split-0 and the one of split-2 are different: the two dataloaders have different batch sizes. Got 1 and 2\n"
            "This may lead to inconsistent results across splits."
        )
        assert len(caplog.records) == 6

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

    def test_on_validate_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.STATE.split_idx = 0
        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-000", "ses-M000")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The validation dataset passed does not contain the same (participant, session) pairs as in the original validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].data_tsv}). "
                "Difference: [('sub-000', 'ses-M000'), ('sub-010', 'ses-M003')]",
            ),
        ):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=self.DATALOADER
            )

        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-010", "ses-M003")])
        self.GOOD_DATASET.to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=self.DATALOADER
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(
            MAPS.training.data.validation.splits[0].dataset_json, overwrite=True
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=self.DATALOADER
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.validate. Beware that differences between datasets could lead to inconsistent results in validation metrics."
        )
        assert caplog.records[1].message == (
            "The validation datasets of split-0 and the one passed to Trainer.validate are different: the two datasets don't have the same label. Got None and age\n"
            "This may lead to inconsistent results in validation metrics."
        )
        assert len(caplog.records) == 2

        write_json(
            MAPS.training.data.validation.splits[0].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_validate_start(
                state=self.STATE, maps=MAPS, dataloader=self.DATALOADER
            )
        assert caplog.records[0].message == (
            f"Could not read the validation dataset of split-0 (in {MAPS.training.data.validation.splits[0].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.validate. Beware that differences between datasets could lead to inconsistent results in validation metrics."
        )
        assert len(caplog.records) == 1

    def test_on_test_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-999", "ses-M999")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The test dataset passed does not contain the same (participant, session) pairs as in the original test dataset of group-X (in {MAPS.test.groups['X'].data_tsv}). "
                "Difference: [('sub-100', 'ses-M000'), ('sub-999', 'ses-M999')]",
            ),
        ):
            self.checker.on_test_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )

        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-100", "ses-M000")])
        self.GOOD_DATASET.to_json(MAPS.test.groups["X"].dataset_json, overwrite=True)
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(MAPS.test.groups["X"].dataset_json, overwrite=True)
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the test dataset of group-X (in {MAPS.test.groups['X'].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.test. Beware that differences between datasets could lead to inconsistent results in test metrics."
        )
        assert caplog.records[1].message == (
            "The test datasets of group-X and the one passed to Trainer.test are different: the two datasets don't have the same label. Got None and age\n"
            "This may lead to inconsistent results in test metrics."
        )
        assert len(caplog.records) == 2

        write_json(
            MAPS.test.groups["X"].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_test_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the test dataset of group-X (in {MAPS.test.groups['X'].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.test. Beware that differences between datasets could lead to inconsistent results in test metrics."
        )
        assert len(caplog.records) == 1

    def test_on_predict_start(self, caplog, tmp_path):
        MAPS = create_new_maps(tmp_path)
        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-999", "ses-M999")])

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The prediction dataset passed does not contain the same (participant, session) pairs as in the original prediction dataset of group-X (in {MAPS.prediction.groups['X'].data_tsv}). "
                "Difference: [('sub-100', 'ses-M000'), ('sub-999', 'ses-M999')]",
            ),
        ):
            self.checker.on_predict_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )

        self.DATALOADER.dataset = self.GOOD_DATASET.subset([("sub-100", "ses-M000")])
        self.GOOD_DATASET.to_json(
            MAPS.prediction.groups["X"].dataset_json, overwrite=True
        )
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert len(caplog.records) == 0

        self.BAD_DATASET.to_json(
            MAPS.prediction.groups["X"].dataset_json, overwrite=True
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the arguments ['transforms'] of the prediction dataset of group-X (in {MAPS.prediction.groups['X'].dataset_json}), "
            "and thus could not compare with the dataset passed to Trainer.predict. Beware that differences between datasets could lead to inconsistent results in predictions."
        )
        assert caplog.records[1].message == (
            "The prediction datasets of group-X and the one passed to Trainer.predict are different: the two datasets don't have the same label. Got None and age\n"
            "This may lead to inconsistent results in predictions."
        )
        assert len(caplog.records) == 2

        write_json(
            MAPS.prediction.groups["X"].dataset_json,
            {"abc": 0},
            overwrite=True,
        )
        caplog.clear()
        with caplog.at_level("WARNING"):
            self.checker.on_predict_start(
                group_name="X",
                maps=MAPS,
                dataloader=self.DATALOADER,
                model_checkpoint=MODEL_CHECKPOINT,
            )
        assert caplog.records[0].message == (
            f"Could not read the prediction dataset of group-X (in {MAPS.prediction.groups['X'].dataset_json}), "
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
                "label",
                "age",
                "the two datasets don't have the same label. Got age and None",
            ),
            (
                "transforms",
                Transforms(extraction=Slice()),
                "the two datasets don't have the same transforms. Got Transforms configuration for slice extraction.*Transforms configuration for image extraction.*",
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


def test_state_dict():
    ChecksCallback().load_state_dict(ChecksCallback().state_dict())
