# coding: utf8
import json
import shutil
from os.path import exists
from pathlib import Path

import pytest

from clinicadl import MapsManager

from .testing_tools import compare_folders, modify_maps


@pytest.fixture(
    params=[
        "predict_image_classification",
        "predict_roi_regression",
        "predict_slice_classification",
        "predict_patch_regression",
        "predict_patch_multi_classification",
        "predict_roi_reconstruction",
    ]
)
def test_name(request):
    return request.param


def test_predict(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "predict" / "in"
    ref_dir = base_dir / "predict" / "ref"
    tmp_out_dir = tmp_path / "predict" / "out"
    tmp_out_dir.mkdir(parents=True)

    if test_name == "predict_image_classification":
        maps_name = "maps_image_cnn"
        modes = ["image"]
        use_labels = True
    elif test_name == "predict_slice_classification":
        maps_name = "maps_slice_cnn"
        modes = ["image", "slice"]
        use_labels = True
    elif test_name == "predict_patch_regression":
        maps_name = "maps_patch_cnn"
        modes = ["image", "patch"]
        use_labels = False
    elif test_name == "predict_roi_regression":
        maps_name = "maps_roi_cnn"
        modes = ["image", "roi"]
        use_labels = False
    elif test_name == "predict_patch_multi_classification":
        maps_name = "maps_patch_multi_cnn"
        modes = ["image", "patch"]
        use_labels = False
    elif test_name == "predict_roi_reconstruction":
        maps_name = "maps_roi_ae"
        modes = ["roi"]
        use_labels = False
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    shutil.copytree(input_dir / maps_name, tmp_out_dir / maps_name)
    model_folder = tmp_out_dir / maps_name

    if cmdopt["adapt-base-dir"]:
        with open(model_folder / "maps.json", "r") as f:
            config = json.load(f)
        config = modify_maps(
            maps=config,
            base_dir=base_dir,
            no_gpu=cmdopt["no-gpu"],
            adapt_base_dir=cmdopt["adapt-base-dir"],
        )
        with open(model_folder / "maps.json", "w") as f:
            json.dump(config, f, skipkeys=True, indent=4)

        with open(model_folder / "groups/test-RANDOM/maps.json", "r") as f:
            config = json.load(f)
        config = modify_maps(
            maps=config,
            base_dir=base_dir,
            no_gpu=False,
            adapt_base_dir=cmdopt["adapt-base-dir"],
        )
        with open(model_folder / "groups/test-RANDOM/maps.json", "w") as f:
            json.dump(config, f, skipkeys=True, indent=4)

    tmp_out_subdir = str(model_folder / "split-0/best-loss/test-RANDOM")
    if exists(tmp_out_subdir):
        shutil.rmtree(tmp_out_subdir)

    # # Correction of JSON file for ROI
    # if "roi" in modes:
    #     json_path = model_folder / "maps.json"
    #     with open(json_path, "r") as f:
    #         parameters = json.load(f)
    #     parameters["roi_list"] = ["leftHippocampusBox", "rightHippocampusBox"]
    #     json_data = json.dumps(parameters, skipkeys=True, indent=4)
    #     with open(json_path, "w") as f:
    #         f.write(json_data)

    maps_manager = MapsManager(model_folder, verbose="debug")
    maps_manager.predict(
        data_group="test-RANDOM",
        caps_directory=input_dir / "caps_random",
        tsv_path=input_dir / "caps_random/data.tsv",
        gpu=False,
        use_labels=use_labels,
        overwrite=True,
        diagnoses=["CN"],
    )

    for mode in modes:
        maps_manager.get_prediction(data_group="test-RANDOM", mode=mode)
        if use_labels:
            maps_manager.get_metrics(data_group="test-RANDOM", mode=mode)

    assert compare_folders(
        tmp_out_dir / maps_name,
        input_dir / maps_name,
        tmp_out_dir,
    )
