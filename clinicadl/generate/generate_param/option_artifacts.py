import click

from clinicadl.generate.generate_config import GenerateArtifactsConfig

config_artifacts = GenerateArtifactsConfig.model_fields

contrast = click.option(
    "--contrast/--no-contrast",
    type=config_artifacts["contrast"].annotation,
    default=config_artifacts["contrast"].default,
    is_flag=True,
    help="",
)
gamma = click.option(
    "--gamma",
    multiple=2,
    type=config_artifacts["gamma"].annotation,
    default=config_artifacts["gamma"].default,
    help="Range between -1 and 1 for gamma augmentation",
)
# Motion
motion = click.option(
    "--motion/--no-motion",
    type=config_artifacts["motion"].annotation,
    default=config_artifacts["motion"].default,
    is_flag=True,
    help="",
)
translation = click.option(
    "--translation",
    multiple=2,
    type=config_artifacts["translation"].annotation,
    default=config_artifacts["translation"].default,
    help="Range in mm for the translation",
)
rotation = click.option(
    "--rotation",
    multiple=2,
    type=config_artifacts["rotation"].annotation,
    default=config_artifacts["rotation"].default,
    help="Range in degree for the rotation",
)
num_transforms = click.option(
    "--num_transforms",
    type=config_artifacts["num_transforms"].annotation,
    default=config_artifacts["num_transforms"].default,
    help="Number of transforms",
)
# Noise
noise = click.option(
    "--noise/--no-noise",
    type=config_artifacts["noise"].annotation,
    default=config_artifacts["noise"].default,
    is_flag=True,
    help="",
)
noise_std = click.option(
    "--noise_std",
    multiple=2,
    type=config_artifacts["noise_std"].annotation,
    default=config_artifacts["noise_std"].default,
    help="Range for noise standard deviation",
)
