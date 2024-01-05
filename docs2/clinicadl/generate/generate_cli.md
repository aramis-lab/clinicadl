# Generate Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Generate](./index.md#generate) /
Generate Cli

> Auto-generated documentation for [clinicadl.generate.generate_cli](../../../clinicadl/generate/generate_cli.py) module.

- [Generate Cli](#generate-cli)
  - [RegistrationOrderGroup](#registrationordergroup)
    - [RegistrationOrderGroup().list_commands](#registrationordergroup()list_commands)
  - [cli](#cli)

## RegistrationOrderGroup

[Show source in generate_cli.py:9](../../../clinicadl/generate/generate_cli.py#L9)

CLI group which lists commands by order or registration.

#### Signature

```python
class RegistrationOrderGroup(click.Group):
    ...
```

### RegistrationOrderGroup().list_commands

[Show source in generate_cli.py:12](../../../clinicadl/generate/generate_cli.py#L12)

#### Signature

```python
def list_commands(self, ctx):
    ...
```



## cli

[Show source in generate_cli.py:16](../../../clinicadl/generate/generate_cli.py#L16)

Generation of synthetic dataset.

#### Signature

```python
@click.group(cls=RegistrationOrderGroup, name="generate", no_args_is_help=True)
def cli() -> None:
    ...
```