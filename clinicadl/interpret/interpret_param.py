from pathlib import Path
from typing import get_args

import click

from clinicadl.predict.predict_config import InterpretConfig

config = InterpretConfig.model_fields

input_maps = click.argument("input_maps_directory", type=config["maps_dir"].annotation)
data_group = click.argument("data_group", type=config["data_group"].annotation)
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
save_nifti = click.option(
    "--save_nifti",
    is_flag=True,
    help="Save the output map(s) in the MAPS in NIfTI format.",
)

#### interpret specific ####
name = click.argument(
    "name",
    type=config["name"].annotation,
)
method = click.argument(
    "method",
    type=click.Choice(
        list(config["method_cls"].annotation)
    ),  # ["gradients", "grad-cam"]
)
level = click.option(
    "--level_grad_cam",
    type=get_args(config["level"].annotation)[0],
    default=config["level"].default,
    help="level of the feature map (after the layer corresponding to the number) chosen for grad-cam.",
    show_default=True,
)
target_node = click.option(
    "--target_node",
    type=config["target_node"].annotation,  # int
    default=config["target_node"].default,  # 0
    help="Which target node the gradients explain. Default takes the first output node.",
    show_default=True,
)
save_individual = click.option(
    "--save_individual",
    is_flag=True,
    help="Save individual saliency maps in addition to the mean saliency map.",
)
overwrite_name = click.option(
    "--overwrite_name",
    "-on",
    is_flag=True,
    help="Overwrite the name if it already exists.",
)
