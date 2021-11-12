import os
import shutil

import pytest

from .testing_tools import compare_folders_with_hashes

output_dir = "results"


@pytest.fixture(
    params=[
        "reproduce_image_classification",
    ]
)
def cli_commands(request):
    split = "0"
    if request.param == "reproduce_image_classification":
        config_json = "data/reproducibility/maps.json"
        hash_dict = "data/reproducibility/hashes_dict.obj"
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return config_json, split, hash_dict


def test_train(cli_commands):
    config_json, split, hash_dict = cli_commands
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    flag_error = not os.system(
        f"clinicadl train from_json {config_json} {output_dir} -s {split}"
    )
    assert flag_error
    compare_folders_with_hashes(output_dir, hash_dict)

    shutil.rmtree(output_dir)
