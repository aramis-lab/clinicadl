import importlib
import os
from logging import getLogger
from pathlib import Path

import toml

from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.maps_manager.maps_manager_utils import (
    change_str_to_path,
    read_json,
    remove_unused_tasks,
)

logger = getLogger("clinicadl")


def hf_hub_is_available():
    return importlib.util.find_spec("huggingface_hub") is not None


def push_to_hf_hub(
    hf_hub_path: str,
    maps_dir: Path,
    model_name: str,
):
    if hf_hub_is_available():
        from huggingface_hub import CommitOperationAdd, HfApi
    else:
        raise ModuleNotFoundError(
            "`huggingface_hub` package must be installed to push your model to the HF hub. "
            "Run `python -m pip install huggingface_hub` and log in to your account with "
            "`huggingface-cli login`."
        )

    model_card_ = """
---
language: en
library_name: clinicadl
tags:
- clinicadl
license: mit
---
"""
    hf_hub_path = "ClinicaDL" if hf_hub_path.lower() == "clinicadl" else hf_hub_path

    config_file = maps_dir / "maps.json"
    n_splits = create_readme(
        config_file=config_file, model_name=model_name, model_card=model_card_
    )
    logger.info(f"Uploading {model_name} model to {hf_hub_path} repo in HF hub...")
    api = HfApi()
    hf_operations = []
    id_ = os.path.join(hf_hub_path, model_name)
    user = api.whoami()
    list_orgs = [x["name"] for x in user["orgs"]]

    if hf_hub_path == "ClinicaDL":
        if "ClinicaDL" not in list_orgs:
            raise ClinicaDLArgumentError(
                "You're not in the ClinicaDL organization on Hugging Face. Please follow the link to request to join the organization: https://huggingface.co/clinicadl-test"
            )
    elif hf_hub_path != user["name"]:
        raise ClinicaDLArgumentError(
            f"You're logged as {user['name']} in Hugging Face and you are trying to push a model under {hf_hub_path} logging."
        )

    tmp_file = "tmp_README.md"
    hf_operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=tmp_file),
        CommitOperationAdd(
            path_in_repo="maps.json", path_or_fileobj=maps_dir / "maps.json"
        ),
    ]

    for split in range(n_splits):
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=str(("split-" + str(split)) + "/best-loss/model.pth.tar"),
                path_or_fileobj=str(
                    maps_dir / ("split-" + str(split)) / "best-loss" / "model.pth.tar"
                ),
            )
        )

    for root, dirs, files in os.walk(maps_dir, topdown=False):
        for name in files:
            hf_operations.append(
                CommitOperationAdd(
                    path_in_repo=str(
                        ("split-" + str(split)) + "/best-loss/model.pth.tar"
                    ),
                    path_or_fileobj=str(
                        maps_dir
                        / ("split-" + str(split))
                        / "best-loss"
                        / "model.pth.tar"
                    ),
                )
            )

    try:
        api.create_commit(
            commit_message=f"Uploading {model_name} in {maps_dir}",
            repo_id=id_,
            operations=hf_operations,
            private=True,
        )
        logger.info(f"Successfully uploaded {model_name} to {maps_dir} repo in HF hub!")

    except:
        from huggingface_hub import create_repo

        repo_name = maps_dir.name
        logger.info(f"Creating {repo_name} in the HF hub since it does not exist...")
        create_repo(repo_id=id_)
        logger.info(f"Successfully created {repo_name} in the HF hub!")

        api.create_commit(
            commit_message=f"Uploading {model_name} in {maps_dir}",
            repo_id=id_,
            operations=hf_operations,
        )

    if Path(tmp_file).exists():
        Path(tmp_file).unlink()


def create_readme(
    config_file: Path = None, model_name: str = "test", model_card: str = None
):
    if not config_file.is_file():
        raise ClinicaDLArgumentError("There is no maps.json file in your repository.")

    clinicadl_root_dir = (Path(__file__) / "../..").resolve()
    config_path = (
        Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
    )
    config_dict = toml.load(config_path)

    train_dict = read_json(config_file)
    train_dict = change_str_to_path(train_dict)

    task = train_dict["network_task"]

    config_dict = remove_unused_tasks(config_dict, task)
    config_dict = change_str_to_path(config_dict)

    file = open("tmp_README.md", "w")
    list_lines = []
    list_lines.append(model_card)
    list_lines.append(f"# Model Card for {model_name}  \n")
    list_lines.append(
        f"This model was trained with ClinicaDL. You can find here all the information.\n"
    )

    list_lines.append(f"## General information  \n")

    if train_dict["multi_cohort"]:
        list_lines.append(
            f"This model was trained on several datasets at the same time.   \n"
        )
    list_lines.append(
        f"This model was trained for **{task}** and the architecture chosen is **{train_dict['architecture']}**.  \n"
    )

    for config_section in config_dict:
        list_lines.append(f"### {config_section}  \n")
        for key in config_dict[config_section]:
            if key == "preprocessing_dict":
                list_lines.append(f"### Preprocessing  \n")
                for key_bis in config_dict[config_section][key]:
                    list_lines.append(
                        f"**{key_bis}**: {config_dict[config_section][key][key_bis]}  \n"
                    )
            else:
                if key in train_dict:
                    config_dict[config_section][key] = train_dict[key]
                    train_dict.pop(key)
                list_lines.append(f"**{key}**: {config_dict[config_section][key]}  \n")
    list_lines.append(f"### Other information  \n")
    for key in train_dict:
        list_lines.append(f"**{key}**: {train_dict[key]}  \n")

    file.writelines(list_lines)
    file.close()
    return config_dict["Cross_validation"]["n_splits"]


def load_from_hf_hub(
    output_maps: Path, hf_hub_path: str, maps_name: str
):  # pragma: no cover
    """Class method to be used to load a pretrained model from the Hugging Face hub

    Parameters
    ----------
    output_path: str,

    hf_hub_path: (str)
        The path where the model should have been be saved on thehugginface hub.
    maps_name: str
    """

    if hf_hub_is_available():
        from huggingface_hub import HfApi, snapshot_download
    else:
        raise ModuleNotFoundError(
            "`huggingface_hub` package must be installed to push your model to the HF hub. "
            "Run `python -m pip install huggingface_hub` and log in to your account with "
            "`huggingface-cli login`."
        )

    hf_hub_path = "ClinicaDL" if hf_hub_path.lower() == "clinicadl" else hf_hub_path

    api = HfApi()
    id_ = os.path.join(hf_hub_path, maps_name)
    user = api.whoami()
    list_orgs = [x["name"] for x in user["orgs"]]

    if hf_hub_path == "ClinicaDL":
        if "ClinicaDL" not in list_orgs:
            raise ClinicaDLArgumentError(
                "You're not in the ClinicaDL organization on Hugging Face. Please follow the link to request to join the organization: https://huggingface.co/clinicadl-test"
            )
    elif hf_hub_path != user["name"]:
        logger.warning(
            f"You're logged as {user['name']} in Hugging Face and you are trying to pull a model from {hf_hub_path}."
        )
    else:
        logger.info(f"Downloading {hf_hub_path} files for rebuilding...")

    environment_json = snapshot_download(repo_id=id_, local_dir=output_maps)
