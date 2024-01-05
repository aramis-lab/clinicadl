# Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Tsvtools](./index.md#tsvtools) /
Cli

> Auto-generated documentation for [clinicadl.tsvtools.cli](../../../clinicadl/tsvtools/cli.py) module.

- [Cli](#cli)
  - [RegistrationOrderGroup](#registrationordergroup)
    - [RegistrationOrderGroup().list_commands](#registrationordergroup()list_commands)
  - [cli](#cli)

## RegistrationOrderGroup

[Show source in cli.py:14](../../../clinicadl/tsvtools/cli.py#L14)

CLI group which lists commands by order or registration.

#### Signature

```python
class RegistrationOrderGroup(click.Group):
    ...
```

### RegistrationOrderGroup().list_commands

[Show source in cli.py:17](../../../clinicadl/tsvtools/cli.py#L17)

#### Signature

```python
def list_commands(self, ctx):
    ...
```



## cli

[Show source in cli.py:21](../../../clinicadl/tsvtools/cli.py#L21)

Manipulation of TSV files to prepare and manage input data.

#### Signature

```python
@click.group(cls=RegistrationOrderGroup, name="tsvtools", no_args_is_help=True)
def cli() -> None:
    ...
```