# Qc Cli

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Quality Check](./index.md#quality-check) /
Qc Cli

> Auto-generated documentation for [clinicadl.quality_check.qc_cli](../../../clinicadl/quality_check/qc_cli.py) module.

- [Qc Cli](#qc-cli)
  - [RegistrationOrderGroup](#registrationordergroup)
    - [RegistrationOrderGroup().list_commands](#registrationordergroup()list_commands)
  - [cli](#cli)

## RegistrationOrderGroup

[Show source in qc_cli.py:8](../../../clinicadl/quality_check/qc_cli.py#L8)

CLI group which lists commands by order or registration.

#### Signature

```python
class RegistrationOrderGroup(click.Group):
    ...
```

### RegistrationOrderGroup().list_commands

[Show source in qc_cli.py:11](../../../clinicadl/quality_check/qc_cli.py#L11)

#### Signature

```python
def list_commands(self, ctx):
    ...
```



## cli

[Show source in qc_cli.py:15](../../../clinicadl/quality_check/qc_cli.py#L15)

Performs quality check procedure for t1-linear or t1-volume pipelines.

Original code can be found at https://github.com/vfonov/deep-qc

#### Signature

```python
@click.group(cls=RegistrationOrderGroup, name="quality-check", no_args_is_help=True)
def cli() -> None:
    ...
```