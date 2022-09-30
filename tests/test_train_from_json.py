import pathlib
import shutil
from os import system
from os.path import join

from .testing_tools import compare_folders_with_hashes, create_hashes_dict, models_equal

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data"


def test_json_compatibility():
    split = "0"
    config_json = pathlib.Path("train_from_json/in/maps_roi_cnn/maps.json")
    output_dir = pathlib.Path("train_from_json/out/maps_reproduced")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    flag_error = not system(
        f"clinicadl train from_json {config_json} {output_dir} -s 0"
    )
    assert flag_error

    shutil.rmtree(output_dir)


def test_determinism():
    input_dir = pathlib.Path("train_from_json/out/maps_roi_cnn")
    output_dir = pathlib.Path("train_from_json/out/reproduced_MAPS")
    if input_dir.exists():
        shutil.rmtree(input_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    test_input = [
        "train",
        "classification",
        "train_from_json/in/caps_roi",
        "t1-linear_mode-roi.json",
        "train_from_json/in/labels_list",
        str(input_dir),
        "-c",
        "train_from_json/in/reproducibility_config.toml",
    ]
    # Run first experiment
    flag_error = not system("clinicadl " + " ".join(test_input))
    assert flag_error
    input_hashes = create_hashes_dict(
        input_dir,
        ignore_pattern_list=["tensorboard", ".log", "training.tsv", "maps.json"],
    )

    # Reproduce experiment
    config_json = input_dir.joinpath("maps.json")
    flag_error = not system(
        f"clinicadl train from_json {config_json} {output_dir} -s 0"
    )
    assert flag_error
    compare_folders_with_hashes(
        output_dir,
        input_hashes,
        ignore_pattern_list=["tensorboard", ".log", "training.tsv", "maps.json"],
    )
    shutil.rmtree(input_dir)
    shutil.rmtree(output_dir)
