import os
import pathlib
import shutil
from os import path, system
from os.path import join
from pathlib import Path

from .testing_tools import compare_folders_with_hashes, create_hashes_dict, models_equal


def test_json_compatibility(cmdopt, tmp_path):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "train_from_json" / "in"
    ref_dir = base_dir / "train_from_json" / "ref"
    tmp_out_dir = tmp_path / "train_from_json" / "out"
    tmp_out_dir.mkdir(parents=True)

    split = "0"
    config_json = os.path.join(input_dir, "maps_roi_cnn/maps.json")
    reproduced_maps_dir = os.path.join(tmp_out_dir, "maps_reproduced")

    if Path(reproduced_maps_dir).exists():
        shutil.rmtree(reproduced_maps_dir)

    flag_error = not system(
        f"clinicadl train from_json {config_json} {reproduced_maps_dir} -s {split}"
    )
    assert flag_error


def test_determinism(cmdopt, tmp_path):

    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "train_from_json" / "in"
    ref_dir = base_dir / "train_from_json" / "ref"
    tmp_out_dir = tmp_path / "train_from_json" / "out"
    tmp_out_dir.mkdir(parents=True)

    maps_dir = path.join(tmp_out_dir, "maps_roi_cnn")
    reproduced_maps_dir = path.join(tmp_out_dir, "reproduced_MAPS")
    if Path(maps_dir).exists():
        shutil.rmtree(maps_dir)
    if Path(reproduced_maps_dir).exists():
        shutil.rmtree(reproduced_maps_dir)
    test_input = [
        "train",
        "classification",
        path.join(str(input_dir), "caps_roi"),
        "t1-linear_mode-roi.json",
        path.join(str(input_dir), "labels_list", "2_fold"),
        str(maps_dir),
        "-c",
        path.join(str(input_dir), "reproducibility_config.toml"),
    ]
    # Run first experiment
    flag_error = not system("clinicadl " + " ".join(test_input))
    assert flag_error
    input_hashes = create_hashes_dict(
        Path(maps_dir),
        ignore_pattern_list=["tensorboard", ".log", "training.tsv", "maps.json"],
    )

    # Reproduce experiment
    config_json = tmp_out_dir.joinpath("maps_roi_cnn/maps.json")
    flag_error = not system(
        f"clinicadl train from_json {config_json} {reproduced_maps_dir} -s 0"
    )
    assert flag_error
    compare_folders_with_hashes(
        Path(reproduced_maps_dir),
        input_hashes,
        ignore_pattern_list=["tensorboard", ".log", "training.tsv", "maps.json"],
    )
