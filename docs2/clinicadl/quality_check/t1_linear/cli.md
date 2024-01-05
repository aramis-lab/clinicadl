# Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Quality Check](../index.md#quality-check) /
[T1 Linear](./index.md#t1-linear) /
Cli

> Auto-generated documentation for [clinicadl.quality_check.t1_linear.cli](../../../../clinicadl/quality_check/t1_linear/cli.py) module.

- [Cli](#cli)
  - [cli](#cli)

## cli

[Show source in cli.py:8](../../../../clinicadl/quality_check/t1_linear/cli.py#L8)

Performs quality check on t1-linear pipeline.

CAPS_DIRECTORY is the CAPS folder where t1-linear outputs are stored.

OUTPUT_TSV is the path to the tsv file where results will be saved.

#### Signature

```python
@click.command(name="t1-linear", no_args_is_help=True)
@cli_param.argument.caps_directory
@click.argument("output_tsv", type=click.Path(path_type=Path))
@cli_param.option.participant_list
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help=(
        "The threshold on the output probability to decide if the image passed or"
        " failed."
    ),
)
@cli_param.option.batch_size
@cli_param.option.n_proc
@cli_param.option.use_gpu
@click.option(
    "--network",
    default="darq",
    type=click.Choice(["darq", "deep_qc", "sq101"]),
    help=(
        "is the architecture chosen for the network (to chose between darq, sq101 and"
        " deep_qc"
    ),
)
@click.option(
    "--use_tensor",
    type=bool,
    default=False,
    is_flag=True,
    help=(
        "Flag allowing the pipeline to run on the extracted tensors and not on the nifti"
        " images"
    ),
)
def cli(
    caps_directory,
    output_tsv,
    participants_tsv,
    threshold,
    batch_size,
    n_proc,
    gpu,
    network,
    use_tensor,
    use_uncropped_image=True,
):
    ...
```