# Generate Random Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Random Cli

> Auto-generated documentation for [clinicadl.generate.generate_random_cli](../../../clinicadl/generate/generate_random_cli.py) module.

- [Generate Random Cli](#generate-random-cli)
  - [cli](#cli)

## cli

[Show source in generate_random_cli.py:6](../../../clinicadl/generate/generate_random_cli.py#L6)

Addition of random gaussian noise to brain images.

CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

GENERATED_CAPS_DIRECTORY is a CAPS folder where the random dataset will be saved.

#### Signature

```python
@click.command(name="random", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.preprocessing
@cli_param.option.participant_list
@cli_param.option.n_subjects
@click.option(
    "--mean",
    type=float,
    default=0,
    help="Mean value of the gaussian noise added to synthetic images.",
)
@click.option(
    "--sigma",
    type=float,
    default=0.5,
    help="Standard deviation of the gaussian noise added to synthetic images.",
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
    mean,
    sigma,
    use_uncropped_image,
    tracer,
    suvr_reference_region,
):
    ...
```