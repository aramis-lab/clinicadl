# Argument

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Cli Param](./index.md#cli-param) /
Argument

> Auto-generated documentation for [clinicadl.utils.cli_param.argument](../../../../clinicadl/utils/cli_param/argument.py) module.

#### Attributes

- `merged_tsv` - ANALYSIS: `click.argument('merged_tsv', type=click.Path(exists=True, path_type=Path))`

- `data_tsv` - TSV TOOLS: `click.argument('data_tsv', type=click.Path(exists=True, path_type=Path))`

- `generated_caps` - GENERATE: `click.argument('generated_caps_directory', type=click.Path(path_type=Path))`

- `data_group` - PREDICT: `click.argument('data_group', type=str)`

- `preprocessing_json` - TRAIN: `click.argument('preprocessing_json', type=str)`

- `modality` - EXTRACT: `click.argument('modality', type=click.Choice(['t1-linear', 'pet-linear', 'custom']))`
- [Argument](#argument)
