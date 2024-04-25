from pathlib import Path
from typing import get_args

import click

from clinicadl import MapsManager
from clinicadl.predict.predict_config import PredictConfig

config = PredictConfig.model_fields

input_maps = click.argument("input_maps_directory", type=config["maps_dir"].annotation)
data_group = click.argument("data_group", type=config["data_group"].annotation)
participants_list = click.option(
    "--participants_tsv",
    type=get_args(config["tsv_path"].annotation)[0],  # Path
    default=config["tsv_path"].default,  # None
    help="""Path to the file with subjects/sessions to process, if different from the one used during network training.
    If it includes the filename will load the TSV file directly.
    Else will load the baseline TSV files of wanted diagnoses produced by `tsvtool split`.""",
    show_default=True,
)
caps_directory = click.option(
    "--caps_directory",
    type=get_args(config["caps_directory"].annotation)[0],  # Path
    default=config["caps_directory"].default,  # None
    help="Data using CAPS structure, if different from the one used during network training.",
    show_default=True,
)
multi_cohort = click.option(
    "--multi_cohort",
    is_flag=True,
    help="Performs multi-cohort interpretation. In this case, caps_directory and tsv_path must be paths to TSV files.",
)
diagnoses = click.option(
    "--diagnoses",
    "-d",
    type=get_args(config["diagnoses"].annotation)[0],  # str list ?
    default=config["diagnoses"].default,  # ??
    multiple=True,
    help="List of diagnoses used for inference. Is used only if PARTICIPANTS_TSV leads to a folder.",
    show_default=True,
)
save_nifti = click.option(
    "--save_nifti",
    is_flag=True,
    help="Save the output map(s) in the MAPS in NIfTI format.",
)
selection_metrics = click.option(
    "--selection_metrics",
    "-sm",
    type=get_args(config["selection_metrics"].annotation)[0],  # str list ?
    default=config["selection_metrics"].default,  # ["loss"]
    multiple=True,
    help="""Allow to select a list of models based on their selection metric. Default will
    only infer the result of the best model selected on loss.""",
    show_default=True,
)
n_proc = click.option(
    "-np",
    "--n_proc",
    type=config["n_proc"].annotation,
    default=config["n_proc"].default,
    show_default=True,
    help="Number of cores used during the task.",
)
gpu = click.option(
    "--gpu/--no-gpu",
    show_default=True,
    default=config["gpu"].default,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
)
batch_size = click.option(
    "--batch_size",
    type=config["batch_size"].annotation,  # int
    default=config["batch_size"].default,  # 8
    show_default=True,
    help="Batch size for data loading.",
)
amp = click.option(
    "--amp/--no-amp",
    default=config["amp"].default,  # false
    help="Enables automatic mixed precision during training and inference.",
    show_default=True,
)
overwrite = click.option(
    "--overwrite",
    "-o",
    is_flag=True,
    help="Will overwrite data group if existing. Please give caps_directory and participants_tsv to"
    " define new data group.",
)


# predict specific
use_labels = click.option(
    "--use_labels/--no_labels",
    show_default=True,
    default=config["use_labels"].default,  # false
    help="Set this option to --no_labels if your dataset does not contain ground truth labels.",
)
label = click.option(
    "--label",
    type=config["label"].annotation,  # str
    default=config["label"].default,  # None
    show_default=True,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
save_tensor = click.option(
    "--save_tensor",
    is_flag=True,
    help="Save the reconstruction output in the MAPS in Pytorch tensor format.",
)
save_latent_tensor = click.option(
    "--save_latent_tensor",
    is_flag=True,
    help="""Save the latent representation of the image.""",
)
skip_leak_check = click.option(
    "--skip_leak_check",
    is_flag=True,
    help="Skip the data leakage check.",
)
split = click.option(
    "--split",
    "-s",
    type=get_args(config["split_list"].annotation)[0],  # list[str]
    default=config["split_list"].default,  # [] ?
    multiple=True,
    show_default=True,
    help="Make inference on the list of given splits. By default, inference is done on all the splits.",
)
