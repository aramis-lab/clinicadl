# Generate Trivial Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Trivial Cli

> Auto-generated documentation for [clinicadl.generate.generate_trivial_cli](../../../clinicadl/generate/generate_trivial_cli.py) module.

- [Generate Trivial Cli](#generate-trivial-cli)
  - [cli](#cli)

## cli

[Show source in generate_trivial_cli.py:8](../../../clinicadl/generate/generate_trivial_cli.py#L8)

Generation of trivial dataset with addition of synthetic brain atrophy.

CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.

#### Signature

```python
@click.command(name="trivial", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.n_subjects
@click.option(
    "--mask_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Path to the extracted masks to generate the two labels. Default will try to"
        " download masks and store them at '~/.cache/clinicadl'."
    ),
)
@click.option(
    "--atrophy_percent", type=float, default=60.0, help="Percentage of atrophy applied."
)
@cli_param.option.use_uncropped_image
@cli_param.option.tracer
@cli_param.option.suvr_reference_region
def cli(
    caps_directory,
    generated_caps_directory,
    preprocessing,
    participants_tsv,
    n_subjects,
    mask_path,
    atrophy_percent,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
):
    ...
```