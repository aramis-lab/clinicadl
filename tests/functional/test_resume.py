"""
Resumes the training interrupted in test_regression.py.

The Trainer is recreated from the saved maps and training is resumed.

This test also checks that we obtain the same results when training is interrupted or not.
"""

import shutil
from pathlib import Path

import pytest

from clinicadl.io import Maps
from clinicadl.train import Trainer

from .test_regression import TwoHeadsRegressionModel, build_callbacks


def _resume(maps_path: Path, base_model_dir: Path):
    model = TwoHeadsRegressionModel(
        base_model=base_model_dir,
        lr_mlp=0.0001,
        lr_age=0.0001,
        lr_sex=0.001,
        lambd=100,
    )
    callbacks = build_callbacks()

    trainer = Trainer.from_maps(
        maps_path=maps_path,
        model=model,
        callbacks=callbacks,
    )
    trainer.resume(split_idx=0)


def _modify_paths(maps_path: Path, bids_dir: Path):
    maps = Maps(maps_path)
    maps.read()

    dataset_json = maps.open_file(maps.training.data.train.splits[0].dataset_json)
    dataset_json["datasets"][0]["bids"]["path"] = str(bids_dir)
    dataset_json["datasets"][1]["bids"]["path"] = str(bids_dir)
    maps.save_file(
        dataset_json, maps.training.data.train.splits[0].dataset_json, overwrite=True
    )

    dataset_json = maps.open_file(maps.training.data.validation.splits[0].dataset_json)
    dataset_json["datasets"][0]["bids"]["path"] = str(bids_dir)
    dataset_json["datasets"][1]["bids"]["path"] = str(bids_dir)
    maps.save_file(
        dataset_json,
        maps.training.data.validation.splits[0].dataset_json,
        overwrite=True,
    )


def _test_resume(
    tmp_path: Path,
    bids_dir: Path,
    ref_interrupted: Path,
    ref_resumed: Path,
    ref_uninterrupted: Path,
    base_model: Path,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    shutil.copytree(ref_interrupted, maps_path, dirs_exist_ok=True)
    _modify_paths(maps_path, bids_dir)

    _resume(maps_path, base_model)

    compare_maps_dir(
        maps_path,
        ref_resumed,
        except_=["callbacks.json"],
    )
    compare_maps_dir(
        maps_path,
        ref_uninterrupted,
        except_=[
            "exec",
            "callbacks.json",
            "training/split-0/summary.log",
            "training/split-0/logs/computational.tsv",
        ],
    )


def test_train(tmp_path, ref_data, bids_dir):
    maps_classif = Maps(ref_data / "maps_test_classification")
    maps_classif.read()
    base_model = maps_classif.training.splits[0].models.final.model_pt
    _test_resume(
        tmp_path=tmp_path,
        bids_dir=bids_dir,
        ref_interrupted=ref_data / "maps_test_regression_interrupted",
        ref_resumed=ref_data / "maps_test_regression_resumed",
        ref_uninterrupted=ref_data / "maps_test_regression_uninterrupted",
        base_model=base_model,
    )


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, bids_dir):
    maps_classif = Maps(ref_data / "maps_test_classification")
    maps_classif.read()
    base_model = maps_classif.training.splits[0].models.final.model_pt
    _test_resume(
        tmp_path=tmp_path,
        bids_dir=bids_dir,
        ref_interrupted=ref_data / "maps_test_regression_interrupted_gpu",
        ref_resumed=ref_data / "maps_test_regression_resumed_gpu",
        ref_uninterrupted=ref_data / "maps_test_regression_uninterrupted_gpu",
        base_model=base_model,
    )
