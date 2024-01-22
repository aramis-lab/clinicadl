from pathlib import Path

import click

from clinicadl.utils import cli_param


@click.command(name="get-labels", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.output_directory
@cli_param.option.diagnoses
@cli_param.option.modality
@cli_param.option.variables_of_interest
@cli_param.option.caps_directory
@click.option(
    "--restriction_tsv",
    help="Path to a TSV file containing the sessions that can be included.",
    type=click.Path(exists=True, path_type=Path),
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
    "--merged_tsv",
    help="Path to a TSV file containing the results of clinica iotools merge-tsv command if different of results_directory/merged.tsv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--missing_mods",
    help="Path to a directory containing the results of clinica iotools missing-modalities command if different of results_directory/missing_mods/",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--remove_unique_session",
    help="This flag allows to remove subjects with a unique session, else they are kept.",
    type=bool,
    default=False,
    is_flag=True,
)
def cli(
    bids_directory,
    output_directory,
    diagnoses,
    modality,
    restriction_tsv,
    variables_of_interest,
    keep_smc,
    missing_mods,
    merged_tsv,
    remove_unique_session,
    caps_directory,
):
    """Get labels in a tsv file.

    This command executes the two following commands:
        - `clinica iotools merge-tsv`
        - `clinica iotools check-missing-modalities`

    BIDS_DIRECTORY is the path to the BIDS directory.
    TSV_DIR is the path of the directory in which the results will be saved.

    Defaults diagnoses are CN and AD.

    Outputs are stored in OUTPUT_TSV.
    """
    from .get_labels import get_labels

    if len(variables_of_interest) == 0:
        variables_of_interest = None

    get_labels(
        bids_directory,
        diagnoses,
        modality=modality,
        restriction_path=restriction_tsv,
        variables_of_interest=variables_of_interest,
        remove_smc=not keep_smc,
        missing_mods=missing_mods,
        merged_tsv=merged_tsv,
        remove_unique_session_=remove_unique_session,
        output_dir=output_directory,
        caps_directory=caps_directory,
    )


if __name__ == "__main__":
    cli()
