import click

from clinicadl.utils import cli_param


@click.command(name="hypometabolic", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.n_proc
@click.option(
    "--dementia",
    "-d",
    type=click.Choice(["ad", "bvftd", "lvppa", "nfvppa", "pca", "svppa"]),
    default="ad",
    help="Path to the extracted masks to generate the two labels. "
    "Default will try to download masks and store them at '~/.cache/clinicadl'.",
)
@click.option(
    "--dementia_percent",
    "-dp",
    type=float,
    default=30.0,
    help="Percentage of dementia applied.",
)
@click.option(
    "--sigma",
    type=int,
    default=5,
    help="sigma ?",
)
@cli_param.option.use_uncropped_image
def cli(
    caps_directory,
    generated_caps_directory,
    participants_tsv,
    n_subjects,
    n_proc,
    sigma,
    dementia,
    dementia_percent,
    use_uncropped_image,
):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_hypometabolic_dataset

    generate_hypometabolic_dataset(
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        preprocessing="pet-linear",
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        n_proc=n_proc,
        dementia=dementia,
        dementia_percent=dementia_percent,
        sigma=sigma,
        uncropped_image=use_uncropped_image,
    )


if __name__ == "__main__":
    cli()
