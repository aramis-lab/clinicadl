import pathlib
import shutil
from os import system

from .testing_tools import compare_folders_with_hashes, create_hashes_dict, models_equal


def test_json_compatibility():
    split = "0"
    config_json = pathlib.Path("data/reproducibility/maps.json")
    output_dir = pathlib.Path("results")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    flag_error = not system(
        f"clinicadl train from_json {config_json} {output_dir} -s {split}"
    )
    assert flag_error

    shutil.rmtree(output_dir)


def test_determinism():
    input_dir = pathlib.Path("results/input_MAPS")
    output_dir = pathlib.Path("results/reproduced_MAPS")
    test_input = [
        "train",
        "classification",
        "data/dataset/random_example",
        "extract_roi.json",
        "data/labels_list",
        str(input_dir),
        "-c",
        "data/reproducibility_config.toml",
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
