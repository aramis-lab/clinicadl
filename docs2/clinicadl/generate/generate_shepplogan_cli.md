# Generate Shepplogan Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Shepplogan Cli

> Auto-generated documentation for [clinicadl.generate.generate_shepplogan_cli](../../../clinicadl/generate/generate_shepplogan_cli.py) module.

- [Generate Shepplogan Cli](#generate-shepplogan-cli)
  - [cli](#cli)

## cli

[Show source in generate_shepplogan_cli.py:6](../../../clinicadl/generate/generate_shepplogan_cli.py#L6)

Random generation of 2D Shepp-Logan phantoms.

Generate a dataset of 2D images at GENERATED_CAPS_DIRECTORY including
3 subtypes based on Shepp-Logan phantom.

#### Signature

```python
@click.command(name="shepplogan", no_args_is_help=True)
@cli_param.argument.generated_caps
@cli_param.option.n_subjects
@cli_param.option.extract_json
@click.option(
    "--image_size", help="Size in pixels of the squared images.", type=int, default=128
)
@click.option(
    "--cn_subtypes_distribution",
    "-csd",
    type=float,
    multiple=3,
    default=(1.0, 0.0, 0.0),
    help="Probability of each subtype to be drawn in CN label.",
)
@click.option(
    "--ad_subtypes_distribution",
    "-asd",
    type=float,
    multiple=3,
    default=(0.05, 0.85, 0.1),
    help="Probability of each subtype to be drawn in AD label.",
)
@click.option(
    "--smoothing/--no-smoothing",
    default=False,
    help="Adds random smoothing to generated data.",
)
def cli(
    generated_caps_directory,
    image_size,
    extract_json,
    ad_subtypes_distribution,
    cn_subtypes_distribution,
    n_subjects,
    smoothing,
):
    ...
```