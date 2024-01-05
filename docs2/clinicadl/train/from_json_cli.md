# From Json Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Train](./index.md#train) /
From Json Cli

> Auto-generated documentation for [clinicadl.train.from_json_cli](../../../clinicadl/train/from_json_cli.py) module.

- [From Json Cli](#from-json-cli)
  - [cli](#cli)

## cli

[Show source in from_json_cli.py:9](../../../clinicadl/train/from_json_cli.py#L9)

Replicate a deep learning training based on a previously created JSON file.
This is particularly useful to retrain random architectures obtained with a random search.

CONFIG_JSON is the path to the JSON file with the configuration of the training procedure.

OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

#### Signature

```python
@click.command(name="from_json", no_args_is_help=True)
@click.argument("config_json", type=click.Path(exists=True, path_type=Path))
@cli_param.argument.output_maps
@click.option(
    "--split",
    "-s",
    type=int,
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
)
def cli(config_json, output_maps_directory, split):
    ...
```