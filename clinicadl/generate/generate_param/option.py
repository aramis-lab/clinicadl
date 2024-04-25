from pathlib import Path
from typing import get_args

import click

from clinicadl.generate.generate_config import (
    Preprocessing,
    SharedGenerateConfigTwo,
    SUVRReferenceRegions,
    Tracer,
)

config = SharedGenerateConfigTwo.model_fields

n_proc = click.option(
    "-np",
    "--n_proc",
    type=config["n_proc"].annotation,
    default=config["n_proc"].default,
    show_default=True,
    help="Number of cores used during the task.",
)
preprocessing = click.option(
    "--preprocessing",
    type=click.Choice(Preprocessing.list()),
    default=config["preprocessing_cls"].default.value,
    required=True,
    help="Preprocessing used to generate synthetic data.",
    show_default=True,
)
participants_tsv = click.option(
    "--participants_tsv",
    type=get_args(config["participants_list"].annotation)[0],
    default=config["participants_list"].default,
    help="Path to a TSV file including a list of participants/sessions.",
    show_default=True,
)
use_uncropped_image = click.option(
    "-uui",
    "--use_uncropped_image",
    is_flag=True,
    help="Use the uncropped image instead of the cropped image generated by t1-linear or pet-linear.",
    show_default=True,
)
tracer = click.option(
    "--tracer",
    type=click.Choice(Tracer.list()),
    default=config["tracer_cls"].default.value,
    help=(
        "Acquisition label if MODALITY is `pet-linear`. "
        "Name of the tracer used for the PET acquisition (trc-<tracer>). "
        "For instance it can be '18FFDG' for fluorodeoxyglucose or '18FAV45' for florbetapir."
    ),
    show_default=True,
)
suvr_reference_region = click.option(
    "-suvr",
    "--suvr_reference_region",
    type=click.Choice(SUVRReferenceRegions.list()),
    default=config["suvr_reference_region_cls"].default.value,
    help=(
        "Regions used for normalization if MODALITY is `pet-linear`. "
        "Intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake "
        "value ratio (SUVR) map. It can be cerebellumPons or cerebellumPon2 (used for amyloid tracers) or pons or "
        "pons2 (used for 18F-FDG tracers)."
    ),
    show_default=True,
)
n_subjects = click.option(
    "--n_subjects",
    type=config["n_subjects"].annotation,
    default=config["n_subjects"].default,
    help="Number of subjects in each class of the synthetic dataset.",
    show_default=True,
)
