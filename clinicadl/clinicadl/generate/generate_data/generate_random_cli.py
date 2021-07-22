import click

from clinicadl.utils import cli_param

cmd_name = "random"


@click.command(name=cmd_name)
@cli_param.argument.caps_directory
@cli_param.argument.participant_list
@cli_param.argument.generated_caps
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
    participant_list,
    generated_caps,
    n_subjects,
    mean,
    sigma,
    preprocessing,
):
    """ """
    from .generate import generate_shepplogan_dataset

    generate_shepplogan_dataset(
        caps_dir=caps_directory,
        tsv_path=participant_list,
        output_dir=generated_caps,
        n_subjects=n_subjects,
        mean=mean,
        sigma=sigma,
        preprocessing=preprocessing,
    )


if __name__ == "__main__":
    cli()
