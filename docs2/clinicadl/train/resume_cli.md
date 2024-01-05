# Resume Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Train](./index.md#train) /
Resume Cli

> Auto-generated documentation for [clinicadl.train.resume_cli](../../../clinicadl/train/resume_cli.py) module.

- [Resume Cli](#resume-cli)
  - [cli](#cli)

## cli

[Show source in resume_cli.py:7](../../../clinicadl/train/resume_cli.py#L7)

Resume training job in specified maps.

INPUT_MAPS_DIRECTORY is the path to the MAPS folder where training job has started.

#### Signature

```python
@click.command(name="resume", no_args_is_help=True)
@cli_param.argument.input_maps
@train_option.split
def cli(input_maps_directory, split):
    ...
```