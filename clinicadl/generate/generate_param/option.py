from pathlib import Path

import click

from clinicadl.generate.generate_config import SharedGenerateConfigTwo

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
    type=config["preprocessing"].annotation,
    default=config["preprocessing"].default,
    required=True,
    help="Preprocessing used to generate synthetic data.",
)
participants_tsv = click.option(
    "--participants_tsv",
    type=config["participants_tsv"].annotation,
    default=config["participants_tsv"].default,
    help="Path to a TSV file including a list of participants/sessions.",
)
use_uncropped_image = click.option(
    "-uui",
    "--use_uncropped_image",
    is_flag=True,
    type=config["use_uncropped_image"].annotation,
    default=config["use_uncropped_image"].default,
    help="Use the uncropped image instead of the cropped image generated by t1-linear or pet-linear.",
)
tracer = click.option(
    "--tracer",
    type=config["tracer"].annotation,
    default=config["tracer"].default,
    help=(
        "Acquisition label if MODALITY is `pet-linear`. "
        "Name of the tracer used for the PET acquisition (trc-<tracer>). "
        "For instance it can be '18FFDG' for fluorodeoxyglucose or '18FAV45' for florbetapir."
    ),
)
suvr_reference_region = click.option(
    "-suvr",
    "--suvr_reference_region",
    type=config["suvr_reference_region"].annotation,
    default=config["suvr_reference_region"].default,
    help=(
        "Regions used for normalization if MODALITY is `pet-linear`. "
        "Intensity normalization using the average PET uptake in reference regions resulting in a standardized uptake "
        "value ratio (SUVR) map. It can be cerebellumPons or cerebellumPon2 (used for amyloid tracers) or pons or "
        "pons2 (used for 18F-FDG tracers)."
    ),
)
n_subjects = click.option(
    "--n_subjects",
    type=config["n_subjects"].annotation,
    default=config["n_subjects"].default,
    help="Number of subjects in each class of the synthetic dataset.",
)
