import importlib
import os
from logging import getLogger
from pathlib import Path

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
    maps_name: str,
    maps_dir: Path,
    model_name: str,
    dataset: [],
    split_list: list = [0],
    loss_list: str = ["best-loss"],
    paper_link: str = None,
):
    if not hf_hub_is_available():
        raise ModuleNotFoundError(
            "`huggingface_hub` package must be installed to push your model to the HF hub. "
            "Run `python -m pip install huggingface_hub` and log in to your account with "
            "`huggingface-cli login`."
        )

    else:
        from huggingface_hub import CommitOperationAdd, HfApi, upload_folder

    if paper_link is not None:
        model_card_ = f"""---
language: en
arxiv: {paper_link}
library_name: clinicadl
tags:
- clinicadl
license: mit
---
"""
    else:
        model_card_ = """---
language: en
library_name: clinicadl
tags:
- clinicadl
license: mit
---
"""
    if hf_hub_path == "clinicadl" or hf_hub_path == "Clinicadl":
        hf_hub_path = "ClinicaDL"

    config_file = maps_dir / "maps.json"
    n_splits, validation = create_readme(
        config_file=config_file, model_name=model_name, model_card=model_card_
    )
    logger.info(f"Uploading {model_name} model to {hf_hub_path} repo in HF hub...")
    api = HfApi()
    hf_operations = []
    id_ = os.path.join(hf_hub_path, model_name)
    user = api.whoami()
    list_orgs = [x["name"] for x in user["orgs"]]
    print(list_orgs)

    if hf_hub_path == "ClinicaDL":
        if "ClinicaDL" not in list_orgs:
            raise ClinicaDLArgumentError(
                "You're not in the ClinicaDL organization on Hugging Face. Please follow the link to request to join the organization: https://huggingface.co/clinicadl-test"
            )
    elif hf_hub_path != user["name"]:
        raise ClinicaDLArgumentError(
            f"You're logged as {user['name']} in Hugging Face and you are trying to push a model under {hf_hub_path} logging."
        )

    hf_operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj="tmp_README.md"),
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
            print(root[(len(str(maps_dir))) :] + "/" + name)

            # print(os.path.join((root[(len(str(maps_dir))):], name)))
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
        )
        logger.info(f"Successfully uploaded {model_name} to {maps_dir} repo in HF hub!")

    except:
        from huggingface_hub import create_repo

        repo_name = os.path.basename(os.path.normpath(maps_dir))
        logger.info(f"Creating {repo_name} in the HF hub since it does not exist...")
        create_repo(repo_id=id_)
        logger.info(f"Successfully created {repo_name} in the HF hub!")

        api.create_commit(
            commit_message=f"Uploading {model_name} in {maps_dir}",
            repo_id=id_,
            operations=hf_operations,
        )
    os.remove("tmp_README.md")


def create_readme(
    config_file: Path = None, model_name: str = "test", model_card: str = None
):
    if not config_file.is_file():
        raise ClinicaDLArgumentError("There is no maps.json file in your repository.")
    import json

    import toml

    clinicadl_root_dir = (Path(__file__) / "../..").resolve()
    config_path = (
        Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
    )
    config_dict = toml.load(config_path)
    train_dict = read_json(config_file)

    task = train_dict["network_task"]
    config_dict = remove_unused_tasks(config_dict, task)
    config_dict = change_str_to_path(config_dict)

    default_dict = dict()
    for config_section in config_dict:
        for key in config_dict[config_section]:
            default_dict[key] = config_dict[config_section][key]

    train_dict = change_str_to_path(train_dict)
    for name in train_dict:
        default_dict[name] = train_dict[name]

    file = open("tmp_README.md", "w")
    list_lines = []
    list_lines.append(model_card)
    list_lines.append(f"# Model Card for {model_name}  \n")
    list_lines.append(
        f"This model was trained with ClinicaDL. You can find here the   \n"
    )

    list_lines.append(f"## General information  \n")

    if default_dict["multi_cohort"]:
        list_lines.append(
            f"This model was trained on several datasets at the same time.   \n"
        )

    list_lines.append(f"## Architecture  \n")
    list_lines.append(
        f"This model was trained for **{default_dict['network_task']}** and the architecture chosen is **{default_dict['architecture']}**.  \n"
    )
    list_lines.append(f"**dropout**: {default_dict['dropout']}  \n")
    list_lines.append(f"**latent_space_size**: {default_dict['latent_space_size']}  \n")
    list_lines.append(f"**feature_size**: {default_dict['feature_size']}  \n")
    list_lines.append(f"**n_conv**: {default_dict['n_conv']}  \n")
    list_lines.append(f"**io_layer_channels**: {default_dict['io_layer_channels']}  \n")
    list_lines.append(f"**recons_weight**: {default_dict['recons_weight']}  \n")
    list_lines.append(f"**kl_weight**: {default_dict['kl_weight']}  \n")
    list_lines.append(f"**normalization**: {default_dict['normalization']}  \n")

    for name in train_dict.keys():
        list_lines.append(f"**{name}**: {train_dict[name]}  \n")
    file.writelines(list_lines)
    file.close()
    return default_dict["n_splits"], default_dict["validation"]


def load_from_hf_hub(
    output_maps: Path, hf_hub_path: str, maps_name: str
):  # pragma: no cover
    """Class method to be used to load a pretrained model from the Hugging Face hub

    Args:
        hf_hub_path (str): The path where the model should have been be saved on the
            hugginface hub.

    .. note::
        This function requires the folder to contain:

        - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

        **or**

        - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
            ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
    """

    if not hf_hub_is_available():
        raise ModuleNotFoundError(
            "`huggingface_hub` package must be installed to load models from the HF hub. "
            "Run `python -m pip install huggingface_hub` and log in to your account with "
            "`huggingface-cli login`."
        )

    else:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    id_ = os.path.join(hf_hub_path, maps_name)
    user = api.whoami()
    list_orgs = [x["name"] for x in user["orgs"]]

    if hf_hub_path == "clinicadl-test":
        if "clinicadl-test" not in list_orgs:
            raise ClinicaDLArgumentError(
                "You're not in the ClinicaDL organization on Hugging Face. Please follow the link to request to join the organization: https://huggingface.co/clinicadl-test"
            )
    elif hf_hub_path != user["name"]:
        raise ClinicaDLArgumentError(
            f"You're logged as {user['name']} in Hugging Face and you are trying to push a model under {hf_hub_path} logging."
        )
    else:
        logger.info(f"Downloading {hf_hub_path} files for rebuilding...")

    environment_json = snapshot_download(repo_id=id_, local_dir=output_maps)
