# Get Progression Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Progression](./index.md#get-progression) /
Get Progression Cli

> Auto-generated documentation for [clinicadl.tsvtools.get_progression.get_progression_cli](../../../../clinicadl/tsvtools/get_progression/get_progression_cli.py) module.

- [Get Progression Cli](#get-progression-cli)
  - [cli](#cli)

## cli

[Show source in get_progression_cli.py:8](../../../../clinicadl/tsvtools/get_progression/get_progression_cli.py#L8)

Get the progression of Alzheimer's disease.

DATA_TSV is the path to the tsv file with columns including ["participants_id", "session_id"]

TIME_HORIZON is the time in months chosen to analyse the stability of the label (default is 36)

Outputs are stored in DATA_TSV.

#### Signature

```python
@click.command(name="get-progression", no_args_is_help=True)
@cli_param.argument.data_tsv
@click.option(
    "--time_horizon",
    help=(
        "Time horizon to analyse stability of the label in the case of a progressive"
        " disease."
    ),
    default=36,
    type=int,
)
def cli(data_tsv, time_horizon):
    ...
```