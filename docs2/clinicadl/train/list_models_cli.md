# List Models Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Train](./index.md#train) /
List Models Cli

> Auto-generated documentation for [clinicadl.train.list_models_cli](../../../clinicadl/train/list_models_cli.py) module.

- [List Models Cli](#list-models-cli)
  - [cli](#cli)

## cli

[Show source in list_models_cli.py:4](../../../clinicadl/train/list_models_cli.py#L4)

Show the list of available models in ClinicaDL.

#### Signature

```python
@click.command(name="list_models")
@click.option(
    "-a",
    "--architecture",
    type=str,
    help="Name of the network for which information will be displayed.",
)
@click.option(
    "-i",
    "--input_size",
    type=str,
    help=(
        "Size of the input image in the shape C@HxW if the image is 2D or C@DxHxW if the"
        " image is 3D."
    ),
)
@click.option(
    "-m",
    "--model_layers",
    type=bool,
    default=False,
    is_flag=True,
    help="Display the detailed Pytorch architecture.",
)
def cli(architecture, input_size, model_layers):
    ...
```