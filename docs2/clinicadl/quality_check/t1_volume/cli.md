# Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Volume](./index.md#t1-volume) /
Cli

> Auto-generated documentation for [clinicadl.quality_check.t1_volume.cli](../../../../clinicadl/quality_check/t1_volume/cli.py) module.

- [Cli](#cli)
  - [cli](#cli)

## cli

[Show source in cli.py:8](../../../../clinicadl/quality_check/t1_volume/cli.py#L8)

Performs quality check on t1-volume pipeline.

CAPS_DIRECTORY is the CAPS folder where t1-volume outputs are stored.

OUTPUT_DIRECTORY is the path to the directory in which TSV files will be written.

GROUP_LABEL is the group associated to the gray matter DARTEL template in CAPS_DIRECTORY.

#### Signature

```python
@click.command(name="t1-volume", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument("output_directory", type=click.Path(path_type=Path))
@click.argument("group_label", type=str)
def cli(caps_directory, output_directory, group_label):
    ...
```