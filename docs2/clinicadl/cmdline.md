# Cmdline

[Clinicadl Index](../README.md#clinicadl-index) /
[Clinicadl](./index.md#clinicadl) /
Cmdline

> Auto-generated documentation for [clinicadl.cmdline](../../clinicadl/cmdline.py) module.

- [Cmdline](#cmdline)
  - [cli](#cli)

## cli

[Show source in cmdline.py:23](../../clinicadl/cmdline.py#L23)

ClinicaDL command line.

For more information please read the doc: https://clinicadl.readthedocs.io/en/latest/ .
Source code is available on GitHub: https://github.com/aramis-lab/clinicaDL .

Do not hesitate to create an issue to report a bug or suggest an improvement.

#### Signature

```python
@click.group(context_settings=CONTEXT_SETTINGS, no_args_is_help=True)
@click.version_option()
@click.option("-v", "--verbose", is_flag=True, help="Verbosity mode.")
def cli(verbose):
    ...
```