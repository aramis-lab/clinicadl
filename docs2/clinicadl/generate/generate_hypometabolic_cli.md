# Generate Hypometabolic Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Hypometabolic Cli

> Auto-generated documentation for [clinicadl.generate.generate_hypometabolic_cli](../../../clinicadl/generate/generate_hypometabolic_cli.py) module.

- [Generate Hypometabolic Cli](#generate-hypometabolic-cli)
  - [cli](#cli)

## cli

[Show source in generate_hypometabolic_cli.py:6](../../../clinicadl/generate/generate_hypometabolic_cli.py#L6)

Generation of trivial dataset with addition of synthetic brain atrophy.

CAPS_DIRECTORY is the CAPS folder from where input brain images will be loaded.

GENERATED_CAPS_DIRECTORY is a CAPS folder where the trivial dataset will be saved.

#### Signature

```python
@click.command(name="hypometabolic", no_args_is_help=True)
@cli_param.argument.caps_directory
@cli_param.argument.generated_caps
@cli_param.option.participant_list
@cli_param.option.n_subjects
@cli_param.option.n_proc
@click.option(
    "--pathology",
    "-p",
    type=click.Choice(["ad", "bvftd", "lvppa", "nfvppa", "pca", "svppa"]),
    default="ad",
    help=(
        "Pathology applied. To chose in the following list: [ad, bvftd, lvppa, nfvppa,"
        " pca, svppa]"
    ),
)
@click.option(
    "--anomaly_degree",
    "-anod",
    type=float,
    default=30.0,
    help="Degrees of hypo-metabolism applied (in percent)",
)
@click.option(
    "--sigma",
    type=int,
    default=5,
    help="It is the parameter of the gaussian filter used for smoothing.",
)
@cli_param.option.use_uncropped_image
def cli(
    caps_directory,
    generated_caps_directory,
    participants_tsv,
    n_subjects,
    n_proc,
    sigma,
    pathology,
    anomaly_degree,
    use_uncropped_image,
):
    ...
```