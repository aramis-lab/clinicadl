# Get Labels Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Labels](./index.md#get-labels) /
Get Labels Cli

> Auto-generated documentation for [clinicadl.tsvtools.get_labels.get_labels_cli](../../../../clinicadl/tsvtools/get_labels/get_labels_cli.py) module.

- [Get Labels Cli](#get-labels-cli)
  - [cli](#cli)

## cli

[Show source in get_labels_cli.py:8](../../../../clinicadl/tsvtools/get_labels/get_labels_cli.py#L8)

Get labels in a tsv file.

This command executes the two following commands:
    - `clinica iotools merge-tsv`
    - `clinica iotools check-missing-modalities`

BIDS_DIRECTORY is the path to the BIDS directory.
RESULTS_TSV is the path (including the name of the file) where the results will be save

Defaults diagnoses are CN and AD.

Outputs are stored in OUTPUT_TSV.

#### Signature

```python
@click.command(name="get-labels", no_args_is_help=True)
@cli_param.argument.bids_directory
@cli_param.argument.results_tsv
@cli_param.option.diagnoses
@cli_param.option.modality
@cli_param.option.variables_of_interest
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
    help=(
        "Path to a TSV file containing the results of clinica iotools merge-tsv command"
        " if different of results_directory/merged.tsv"
    ),
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--missing_mods",
    help=(
        "Path to a directory containing the results of clinica iotools"
        " missing-modalities command if different of results_directory/missing_mods/"
    ),
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--remove_unique_session",
    help=(
        "This flag allows to remove subjects with a unique session, else they are kept."
    ),
    type=bool,
    default=False,
    is_flag=True,
)
def cli(
    bids_directory,
    results_tsv,
    diagnoses,
    modality,
    restriction_tsv,
    variables_of_interest,
    keep_smc,
    missing_mods,
    merged_tsv,
    remove_unique_session,
):
    ...
```