import click

from clinicadl.utils import cli_param


@click.command(name="trivial")
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.preprocessing
@click.option(
    "--mask_path",
    type=str,
    default=None,
    help="path to the extracted masks to generate the two labels.",
)
@click.option(
    "--atrophy_percent",
    type=float,
    default=60.0,
    help="Percentage of atrophy applied",
)
def cli(
    caps_directory,
    generated_caps_directory,
    participants_tsv,
    n_subjects,
    preprocessing,
    mask_path,
    atrophy_percent,
):
    """Generation of trivial dataset with addition of synthetic brain atrophy.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.
    """
    from .generate import generate_trivial_dataset

    generate_trivial_dataset(
        caps_dir=caps_directory,
        tsv_path=participants_tsv,
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        preprocessing=preprocessing,
        mask_path=mask_path,
        atrophy_percent=atrophy_percent,
    )


if __name__ == "__main__":
    cli()
