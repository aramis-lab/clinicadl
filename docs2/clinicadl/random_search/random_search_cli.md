# Random Search Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Random Search](./index.md#random-search) /
Random Search Cli

> Auto-generated documentation for [clinicadl.random_search.random_search_cli](../../../clinicadl/random_search/random_search_cli.py) module.

- [Random Search Cli](#random-search-cli)
  - [cli](#cli)

## cli

[Show source in random_search_cli.py:6](../../../clinicadl/random_search/random_search_cli.py#L6)

Hyperparameter exploration using random search.

LAUNCH_DIRECTORY is the path to the parents folder where results of random search will be saved.

NAME is the name of the output folder containing the experiment.

#### Signature

```python
@click.command("random-search", no_args_is_help=True)
@click.argument("launch_directory", type=click.Path(exists=True, path_type=Path))
@click.argument("name", type=str)
def cli(launch_directory, name):
    ...
```