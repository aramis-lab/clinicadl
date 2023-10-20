import importlib
import os
import shutil
import tempfile
import warnings
from logging import getLogger
from pathlib import Path

import cloudpickle
import torch
from torch import nn

logger = getLogger("clinicadl")


model_card_template = """---
language: en
tags:
- clinicadl
license: MIT
---
"""


def push_to_hf_hub(
    hf_hub_path: str,
    maps_dir: Path,
    model_name: str,
    split_list: list = [0],
    loss_list: str = ["best-loss"],
):  # pragma: no cover
    """Method allowing to save your model directly on the huggung face hub.
    You will need to have the `huggingface_hub` package installed and a valid Hugging Face
    account. You can install the package using

    .. code-block:: bash

        python -m pip install huggingface_hub

    end then login using

    .. code-block:: bash

        huggingface-cli login

    Args:
        hf_hub_path (str): path to your repo on the Hugging Face hub.
    """
    if not hf_hub_is_available():
        raise ModuleNotFoundError(
            "`huggingface_hub` package must be installed to push your model to the HF hub. "
            "Run `python -m pip install huggingface_hub` and log in to your account with "
            "`huggingface-cli login`."
        )

    else:
        from huggingface_hub import CommitOperationAdd, HfApi, upload_folder

    logger.info(f"Uploading {model_name} model to {hf_hub_path} repo in HF hub...")

    # tempdir = tempfile.mkdtemp()

    # network.save(tempdir)

    # model_files = os.listdir(maps_dir)

    api = HfApi()
    hf_operations = []

    id_ = hf_hub_path
    api.create_repo(id_, token="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")

    api.upload_folder(
        folder_path=str(maps_dir),
        # path_in_repo="my-dataset/train", # Upload to a specific folder
        repo_id=hf_hub_path,
        repo_type="model",
    )

    # for split in split_list:
    #     hf_operations.append(
    #         CommitOperationAdd(
    #             path_in_repo=str(("split-" + str(split)) + "_model5.pth.tar"),
    #             path_or_fileobj=str(
    #                 maps_dir
    #                 / ("split-" + str(split))
    #                 / loss_list[split]
    #                 / "model.pth.tar"
    #             ),
    #         )
    #     )

    # for file in ["maps.json", "environment.txt", "information.log"]:
    #     hf_operations.append(
    #         CommitOperationAdd(
    #             path_in_repo=file,
    #             path_or_fileobj=str(maps_dir / file),
    #         )
    #     )

    # try:
    #     api.create_commit(
    #         commit_message=f"Uploading {model_name} in {maps_dir}",
    #         repo_id=id_,
    #         operations=hf_operations,
    #     )
    #     logger.info(f"Successfully uploaded {model_name} to {maps_dir} repo in HF hub!")

    # except:
    #     from huggingface_hub import create_repo

    #     repo_name = os.path.basename(os.path.normpath(maps_dir))
    #     logger.info(f"Creating {repo_name} in the HF hub since it does not exist...")
    #     create_repo(repo_id=id_)
    #     logger.info(f"Successfully created {repo_name} in the HF hub!")

    #     api.create_commit(
    #         commit_message=f"Uploading {model_name} in {maps_dir}",
    #         repo_id=id_,
    #         operations=hf_operations,
    #     )


def save_model(network: nn.Module, dir_path: str):
    """Method to save the model at a specific location. It saves, the model weights as a
    ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
    model to save used custom encoder (resp. decoder) provided by the user, these are also
    saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

    Args:
    dir_path (str): The path where the model should be saved. If the path
            path does not exist a folder will be created at the provided location.
    """

    env_spec = EnvironmentConfig(
        python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
    )
    model_dict = {"model_state_dict": deepcopy(network.state_dict())}

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)

        except FileNotFoundError as e:
            raise e

    env_spec.save_json(dir_path, "environment")
    network.model_config.save_json(dir_path, "model_config")

    # only save .pkl if custom architecture provided
    if not network.model_config.uses_default_encoder:
        with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
            cloudpickle.register_pickle_by_value(inspect.getmodule(network.encoder))
            cloudpickle.dump(network.encoder, fp)

    if not network.model_config.uses_default_decoder:
        with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
            cloudpickle.register_pickle_by_value(inspect.getmodule(network.decoder))
            cloudpickle.dump(network.decoder, fp)

    torch.save(model_dict, os.path.join(dir_path, "model.pt"))


def hf_hub_is_available():
    return importlib.util.find_spec("huggingface_hub") is not None


def load_from_hf_hub(
    output_maps: Path, hf_hub_path: str, allow_pickle=False
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
        from huggingface_hub import hf_hub_download

    logger.info(f"Downloading {hf_hub_path} files for rebuilding...")

    environment_json = hf_hub_download(
        repo_id=hf_hub_path, filename="maps.json", local_dir=output_maps
    )
    print(environment_json)
    # #model_config_json = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")

    # _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

    # model_config = cls._load_model_config_from_folder(dir_path)
    # dir_path = os.path.dirname(config_path)
    # if (
    #     cls.__name__ + "Config" != model_config.name
    #     and cls.__name__ + "_Config" != model_config.name
    # ):
    #     warnings.warn(
    #         f"You are trying to load a "
    #         f"`{ cls.__name__}` while a "
    #         f"`{model_config.name}` is given."
    #     )

    # model_weights = cls._load_model_weights_from_folder(dir_path)

    # if (
    #     not model_config.uses_default_encoder or not model_config.uses_default_decoder
    # ) and not allow_pickle:
    #     warnings.warn(
    #         "You are about to download pickled files from the HF hub that may have "
    #         "been created by a third party and so could potentially harm your computer. If you "
    #         "are sure that you want to download them set `allow_pickle=true`."
    #     )

    # else:
    #     if not model_config.uses_default_encoder:
    #         _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
    #         encoder = cls._load_custom_encoder_from_folder(dir_path)

    #     else:
    #         encoder = None

    #     if not model_config.uses_default_decoder:
    #         _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
    #         decoder = cls._load_custom_decoder_from_folder(dir_path)

    #     else:
    #         decoder = None

    #     logger.info(f"Successfully downloaded {cls.__name__} model!")

    #     model = cls(model_config, encoder=encoder, decoder=decoder)
    #     model.load_state_dict(model_weights)

    #     return model
