import click

from clinicadl.utils import cli_param


@click.command(name="random")
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.preprocessing
@click.option(
    "--mean",
    type=float,
    default=0,
    help="Mean value of the noise added for the random dataset.",
)
@click.option(
    "--sigma",
    type=float,
    default=0.5,
    help="Standard deviation of the noise added for the random dataset.",
)
def cli(
    caps_directory,
    generated_caps_directory,
    participants_tsv,
    n_subjects,
    mean,
    sigma,
    preprocessing,
):
    """Addition of random gaussian noise to brain images.

    CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.
    GENERATED_CAPS_DIRECTORY is a CAPS folder where the random dataset will be saved.
    """
    from .generate import generate_random_dataset

    generate_random_dataset(
        caps_dir=caps_directory,
        tsv_path=participants_tsv,
        output_dir=generated_caps_directory,
        n_subjects=n_subjects,
        mean=mean,
        sigma=sigma,
        preprocessing=preprocessing,
    )


if __name__ == "__main__":
    cli()
