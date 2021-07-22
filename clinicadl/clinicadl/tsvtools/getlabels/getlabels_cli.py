import click

from clinicadl.utils import cli_param

cmd_name = "getlabels"


@click.command(name=cmd_name)
@cli_param.argument.merged_tsv
@cli_param.argument.missing_mods_directory
@cli_param.argument.results_directory
@cli_param.option.diagnoses
@cli_param.option.modality
@click.option(
    "--time_horizon",
    help="Time horizon to analyse stability of MCI subjects.",
    default=36,
    type=int,
)
@click.option(
    "--restriction_path",
    help="Path to a tsv containing the sessions that can be included.",
    type=str,
    default=None,
)
@click.option(
    "--variables_of_interest",
    help="Variables of interest that will be kept in the final lists."
    "Default will keep the diagnosis, age and the sex needed for the split procedure.",
    type=str,
    multiple=True,
    default=None,
)
@click.option(
    "--keep_smc",
    help="This flag allows to keep SMC participants, else they are removed.",
    type=bool,
    default=False,
    is_flag=True,
)
def cli(
    merged_tsv,
    missing_mods_directory,
    results_directory,
    diagnoses,
    modality,
    restriction_path,
    time_horizon,
    variables_of_interest,
    remove_smc,
):
    """ """
    # import function to execute
    from .getlabels import get_labels

    # run function
    get_labels(
        merged_tsv,
        missing_mods_directory,
        results_directory,
        diagnoses,
        modality=modality,
        restriction_path=restriction_path,
        time_horizon=time_horizon,
        variables_of_interest=variables_of_interest,
        remove_smc=remove_smc,
    )


if __name__ == "__main__":
    cli()
