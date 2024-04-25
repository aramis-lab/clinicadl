from typing import get_args

import click

from clinicadl.generate.generate_config import GenerateArtifactsConfig

config_artifacts = GenerateArtifactsConfig.model_fields

contrast = click.option(
    "--contrast/--no-contrast",
    default=config_artifacts["contrast"].default,
    help="",
    show_default=True,
)
gamma = click.option(
    "--gamma",
    multiple=2,
    type=get_args(config_artifacts["gamma"].annotation)[0],
    default=config_artifacts["gamma"].default,
    help="Range between -1 and 1 for gamma augmentation",
    show_default=True,
)
# Motion
motion = click.option(
    "--motion/--no-motion",
    default=config_artifacts["motion"].default,
    help="",
    show_default=True,
)
translation = click.option(
    "--translation",
    multiple=2,
    type=get_args(config_artifacts["translation"].annotation)[0],
    default=config_artifacts["translation"].default,
    help="Range in mm for the translation",
    show_default=True,
)
rotation = click.option(
    "--rotation",
    multiple=2,
    type=get_args(config_artifacts["rotation"].annotation)[0],
    default=config_artifacts["rotation"].default,
    help="Range in degree for the rotation",
    show_default=True,
)
num_transforms = click.option(
    "--num_transforms",
    type=config_artifacts["num_transforms"].annotation,
    default=config_artifacts["num_transforms"].default,
    help="Number of transforms",
    show_default=True,
)
# Noise
noise = click.option(
    "--noise/--no-noise",
    default=config_artifacts["noise"].default,
    help="",
    show_default=True,
)
noise_std = click.option(
    "--noise_std",
    multiple=2,
    type=get_args(config_artifacts["noise_std"].annotation)[0],
    default=config_artifacts["noise_std"].default,
    help="Range for noise standard deviation",
    show_default=True,
)
