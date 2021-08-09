import click

from clinicadl.utils import cli_param

cmd_name = "random"


@click.command(name=cmd_name)
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
    input_caps_directory,
    output_caps_directory,
    participants_tsv,
    n_subjects,
    mean,
    sigma,
    preprocessing,
):
    """
    Generate a random dataset OUTPUT_CAPS_DIRECTORY in which gaussian noise is added
    to brain images of INPUT_CAPS_DIRECTORY.
    """
    from .generate import generate_random_dataset

    generate_random_dataset(
        caps_dir=input_caps_directory,
        tsv_path=participants_tsv,
        output_dir=output_caps_directory,
        n_subjects=n_subjects,
        mean=mean,
        sigma=sigma,
        preprocessing=preprocessing,
    )


if __name__ == "__main__":
    cli()
