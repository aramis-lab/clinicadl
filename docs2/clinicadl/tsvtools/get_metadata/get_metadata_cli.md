# Get Metadata Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Metadata](./index.md#get-metadata) /
Get Metadata Cli

> Auto-generated documentation for [clinicadl.tsvtools.get_metadata.get_metadata_cli](../../../../clinicadl/tsvtools/get_metadata/get_metadata_cli.py) module.

- [Get Metadata Cli](#get-metadata-cli)
  - [cli](#cli)

## cli

[Show source in get_metadata_cli.py:6](../../../../clinicadl/tsvtools/get_metadata/get_metadata_cli.py#L6)

Writes additional data in the tsv file.

DATA_TSV is the path to the TSV file with colmuns including ["participants_id", "session_id"]

MERGED_TSV is the path to the TSV file with all the data (output of clinica merge-tsv/ clinicadl get-labels)

VARIABLES_OF_INTEREST is a list of variables (columns) that will be added to the tsv file

Outputs are written in DATA_TSV.

#### Signature

```python
@click.command(name="get-metadata", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.argument.merged_tsv
@cli_param.option.variables_of_interest
def cli(data_tsv, merged_tsv, variables_of_interest):
    ...
```