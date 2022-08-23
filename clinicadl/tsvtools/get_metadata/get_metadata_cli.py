from typing import List

import click

from clinicadl.utils import cli_param


@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.results_directory
@cli_param.option.diagnoses
@cli_param.option.modality
@click.option(
    "--time_horizon",
    help="Time horizon to analyse stability of subjects.",
    default=36,
    type=int,
)
@click.option(
    "--restriction_tsv",
    help="Path to a TSV file containing the sessions that can be included.",
    type=str,
    default=None,
)
@click.option(
    "--variables_of_interest",
    help="Variables of interest that will be kept in the final lists. "
    "Will always keep the group, subgroup, age and sex needed for the split procedure.",
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
@click.option(
    "--caps_directory",
    "-c",
    help="input folder of a CAPS compliant dataset",
    type=str,
    default=None,
)
@cli_param.option.stability_dict
def cli(
    bids_directory,
    results_directory,
    diagnoses,
    modality,
    restriction_tsv,
    time_horizon,
    variables_of_interest,
    keep_smc,
    caps_directory,
    stability_dict,
):
    """Get labels in one tsv files.

    This command will perform two others command inside :
        - `clinica iotools merge-tsv`
        - `clinica iotools check-missing-modalities`

    BIDS_DIRECTORY is the path of the BIDS.

    Outputs are stored in RESULTS_TSV.
    """
    from .get_metadata import get_metadata

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    get_metadata(
        bids_directory,
        results_directory,
        diagnoses,
        stability_dict=stability_dict,
        modality=modality,
        restriction_path=restriction_tsv,
        time_horizon=time_horizon,
        variables_of_interest=variables_of_interest,
        remove_smc=not keep_smc,
        caps_directory=caps_directory,
    )


if __name__ == "__main__":
    cli()
