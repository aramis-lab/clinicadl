import click

from clinicadl.generate.generate import generate_artifacts_dataset
from clinicadl.utils import cli_param


@click.command(name="artifacts", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.n_proc
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
##############
# Contrast
@click.option(
    "--contrast/--no-contrast",
    type=bool,
    default=False,
    is_flag=True,
    help="",
)
@click.option(
    "--gamma",
    type=float,
    multiple=2,
    default=[-0.2, -0.05],
    help="Range between -1 and 1 for gamma augmentation",
)
# Motion
@click.option(
    "--motion/--no-motion",
    type=bool,
    default=False,
    is_flag=True,
    help="",
)
@click.option(
    "--translation",
    type=float,
    multiple=2,
    default=[2, 4],
    help="Range in mm for the translation",
)
@click.option(
    "--rotation",
    #    type=float,
    multiple=2,
    default=[2, 4],
    help="Range in degree for the rotation",
)
@click.option(
    "--num_transforms",
    type=int,
    default=2,
    help="Number of transforms",
)
# Noise
@click.option(
    "--noise/--no-noise",
    type=bool,
    default=False,
    is_flag=True,
    help="",
)
@click.option(
    "--noise_std",
    type=float,
    multiple=2,
    default=[5, 15],
    help="Range for noise standard deviation",
)
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
    contrast,
    gamma,
    motion,
    translation,
    rotation,
    num_transforms,
    noise,
    noise_std,
    n_proc,
):
    """Generation of trivial dataset with addition of synthetic artifacts.
    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """

    generate_artifacts_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing=preprocessing,
        output_dir=generated_caps_directory,
        uncropped_image=use_uncropped_image,
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
        contrast=contrast,
        gamma=gamma,
        motion=motion,
        translation=translation,
        rotation=rotation,
        num_transforms=num_transforms,
        noise=noise,
        noise_std=noise_std,
        n_proc=n_proc,
    )


if __name__ == "__main__":
    cli()
