import click

from clinicadl.generate.generate_config import GenerateArtifactsConfig
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

contrast = click.option(
    "--contrast/--no-contrast",
    default=get_default("contrast", GenerateArtifactsConfig),
    help="",
    show_default=True,
)
gamma = click.option(
    "--gamma",
    multiple=2,
    type=get_type("gamma", GenerateArtifactsConfig),
    default=get_default("gamma", GenerateArtifactsConfig),
    help="Range between -1 and 1 for gamma augmentation",
    show_default=True,
)
# Motion
motion = click.option(
    "--motion/--no-motion",
    default=get_default("motion", GenerateArtifactsConfig),
    help="",
    show_default=True,
)
translation = click.option(
    "--translation",
    multiple=2,
    type=get_type("translation", GenerateArtifactsConfig),
    default=get_default("translation", GenerateArtifactsConfig),
    help="Range in mm for the translation",
    show_default=True,
)
rotation = click.option(
    "--rotation",
    multiple=2,
    type=get_type("rotation", GenerateArtifactsConfig),
    default=get_default("rotation", GenerateArtifactsConfig),
    help="Range in degree for the rotation",
    show_default=True,
)
num_transforms = click.option(
    "--num_transforms",
    type=get_type("num_transforms", GenerateArtifactsConfig),
    default=get_default("num_transforms", GenerateArtifactsConfig),
    help="Number of transforms",
    show_default=True,
)
# Noise
noise = click.option(
    "--noise/--no-noise",
    default=get_default("noise", GenerateArtifactsConfig),
    help="",
    show_default=True,
)
noise_std = click.option(
    "--noise_std",
    multiple=2,
    type=get_type("noise_std", GenerateArtifactsConfig),
    default=get_default("noise_std", GenerateArtifactsConfig),
    help="Range for noise standard deviation",
    show_default=True,
)
