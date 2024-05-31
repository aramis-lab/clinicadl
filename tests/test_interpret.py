# coding: utf8

import os
import shutil
from pathlib import Path

import pytest

from clinicadl.config.config.pipelines.interpret import InterpretConfig
from clinicadl.predict.predict_manager import PredictManager


@pytest.fixture(params=["classification", "regression"])
def test_name(request):
    return request.param


def test_interpret(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "interpret" / "in"
    ref_dir = base_dir / "interpret" / "ref"
    tmp_out_dir = tmp_path / "interpret" / "out"
    tmp_out_dir.mkdir(parents=True)

    labels_dir_str = str(input_dir / "labels_list" / "2_fold")
    maps_tmp_out_dir = str(tmp_out_dir / "maps")
    if test_name == "classification":
        cnn_input = [
            "train",
            "classification",
            str(input_dir / "caps_image"),
            "t1-linear_mode-image.json",
            labels_dir_str,
            maps_tmp_out_dir,
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]

    elif test_name == "regression":
        cnn_input = [
            "train",
            "regression",
            str(input_dir / "caps_patch"),
            "t1-linear_mode-patch.json",
            labels_dir_str,
            maps_tmp_out_dir,
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    run_interpret(cnn_input, tmp_out_dir, ref_dir)


def run_interpret(cnn_input, tmp_out_dir, ref_dir):
    from clinicadl.utils.enum import InterpretationMethod

    maps_path = tmp_out_dir / "maps"
    if maps_path.is_dir():
        shutil.rmtree(maps_path)

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    assert train_error

    for method in list(InterpretationMethod):
        interpret_config = InterpretConfig(
            maps_dir=maps_path,
            data_group="train",
            name=f"test-{method}",
            method_cls=method,
        )
        interpret_manager = PredictManager(interpret_config)
        interpret_manager.interpret()
        interpret_map = interpret_manager.get_interpretation(
            "train", f"test-{interpret_config.method}"
        )
