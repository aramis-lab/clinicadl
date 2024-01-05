# Option

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Utils](../index.md#utils) /
[Cli Param](./index.md#cli-param) /
Option

> Auto-generated documentation for [clinicadl.utils.cli_param.option](../../../../clinicadl/utils/cli_param/option.py) module.

#### Attributes

- `diagnoses` - TSV TOOLS: `click.option('--diagnoses', '-d', multiple=True, default=('AD', 'CN'), help="Labels selected for the demographic analysis used in the context of Alzheimer's Disease classification.")`

- `participant_list` - GENERATE: `click.option('--participants_tsv', type=click.Path(exists=True, path_type=Path), help='Path to a TSV file including a list of participants/sessions.')`

- `use_gpu` - Computational: `click.option('--gpu/--no-gpu', default=True, help='Use GPU by default. Please specify `--no-gpu` to force using CPU.')`

- `save_features` - Extract: `click.option('--save_features', type=bool, default=False, is_flag=True, help='Extract the selected mode to save the tensor. By default, the pipeline only save images and the mode extraction\n            is done when images are loaded in the train.')`

- `overwrite` - Data group: `click.option('--overwrite', '-o', default=False, is_flag=True, help='Will overwrite data group if existing. Please give caps_directory and partcipants_tsv to define new data group.')`
- [Option](#option)
