import importlib
import os
import shutil
import tempfile

logger = get_logger("clinicadl")


model_card_template = """---
language: en
tags:
- clinicadl
license: MIT
---
"""


def push_to_hf_hub(self, hf_hub_path: str):  # pragma: no cover
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
        from huggingface_hub import CommitOperationAdd, HfApi

    logger.info(f"Uploading {self.model_name} model to {hf_hub_path} repo in HF hub...")

    tempdir = tempfile.mkdtemp()

    self.save(tempdir)

    model_files = os.listdir(tempdir)

    api = HfApi()
    hf_operations = []

    for file in model_files:
        hf_operations.append(
            CommitOperationAdd(
                path_in_repo=file,
                path_or_fileobj=f"{str(os.path.join(tempdir, file))}",
            )
        )

    with open(os.path.join(tempdir, "model_card.md"), "w") as f:
        f.write(model_card_template)

    hf_operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=os.path.join(tempdir, "model_card.md"),
        )
    )

    try:
        api.create_commit(
            commit_message=f"Uploading {self.model_name} in {hf_hub_path}",
            repo_id=hf_hub_path,
            operations=hf_operations,
        )
        logger.info(
            f"Successfully uploaded {self.model_name} to {hf_hub_path} repo in HF hub!"
        )

    except:
        from huggingface_hub import create_repo

        repo_name = os.path.basename(os.path.normpath(hf_hub_path))
        logger.info(f"Creating {repo_name} in the HF hub since it does not exist...")
        create_repo(repo_id=repo_name)
        logger.info(f"Successfully created {repo_name} in the HF hub!")

        api.create_commit(
            commit_message=f"Uploading {self.model_name} in {hf_hub_path}",
            repo_id=hf_hub_path,
            operations=hf_operations,
        )

    shutil.rmtree(tempdir)


def hf_hub_is_available():
    return importlib.util.find_spec("huggingface_hub") is not None


def load_from_hf_hub(cls, hf_hub_path: str, allow_pickle=False):  # pragma: no cover
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

    logger.info(f"Downloading {cls.__name__} files for rebuilding...")

    _ = hf_hub_download(repo_id=hf_hub_path, filename="environment.json")
    config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
    dir_path = os.path.dirname(config_path)

    _ = hf_hub_download(repo_id=hf_hub_path, filename="model.pt")

    model_config = cls._load_model_config_from_folder(dir_path)

    if (
        cls.__name__ + "Config" != model_config.name
        and cls.__name__ + "_Config" != model_config.name
    ):
        warnings.warn(
            f"You are trying to load a "
            f"`{ cls.__name__}` while a "
            f"`{model_config.name}` is given."
        )

    model_weights = cls._load_model_weights_from_folder(dir_path)

    if (
        not model_config.uses_default_encoder or not model_config.uses_default_decoder
    ) and not allow_pickle:
        warnings.warn(
            "You are about to download pickled files from the HF hub that may have "
            "been created by a third party and so could potentially harm your computer. If you "
            "are sure that you want to download them set `allow_pickle=true`."
        )

    else:

        if not model_config.uses_default_encoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pkl")
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pkl")
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None

        logger.info(f"Successfully downloaded {cls.__name__} model!")

        model = cls(model_config, encoder=encoder, decoder=decoder)
        model.load_state_dict(model_weights)

        return model
