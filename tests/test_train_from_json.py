import os
import shutil

from .testing_tools import compare_folders_with_hashes, create_hashes_dict, models_equal


def test_json_compatibility():
    split = "0"
    config_json = "data/reproducibility/maps.json"
    output_dir = "results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    flag_error = not os.system(
        f"clinicadl train from_json {config_json} {output_dir} -s {split}"
    )
    assert flag_error

    shutil.rmtree(output_dir)


def test_determinism():
    input_dir = "results/input_MAPS"
    output_dir = "results/reproduced_MAPS"
    test_input = [
        "train",
        "classification",
        "data/dataset/random_example",
        "extract_roi.json",
        "data/labels_list",
        input_dir,
        "-c",
        "data/reproducibility_config.toml",
    ]
    # Run first experiment
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    input_hashes = create_hashes_dict(
        input_dir,
        ignore_pattern_list=["tensorboard", ".log", "training.tsv", "maps.json"],
    )

    # Reproduce experiment
    config_json = os.path.join(input_dir, "maps.json")
    flag_error = not os.system(
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
