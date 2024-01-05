# Adapt Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Adapt](./index.md#adapt) /
Adapt Cli

> Auto-generated documentation for [clinicadl.tsvtools.adapt.adapt_cli](../../../../clinicadl/tsvtools/adapt/adapt_cli.py) module.

- [Adapt Cli](#adapt-cli)
  - [cli](#cli)

## cli

[Show source in adapt_cli.py:8](../../../../clinicadl/tsvtools/adapt/adapt_cli.py#L8)

Converts split and kfold directories created with clinicaDL 1.1.1 and earlier version
to the last version.

OLD_TSV_DIR is the output directory of the split/kfold pipeline that contains all the TSV files with clinicaDL 1.1.1 and earlier versions.

Results are stored in NEW_TSV_DIR.

#### Signature

```python
@click.command(name="adapt", no_args_is_help=True)
@cli_param.argument.old_tsv_dir
@cli_param.argument.new_tsv_dir
@click.option(
    "--labels_list",
    "-lb",
    type=str,
    multiple=True,
    help="Labels used to create the tsv directory in the old way",
)
def cli(old_tsv_dir, new_tsv_dir, labels_list=None):
    ...
```