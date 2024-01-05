# Train Option

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Train](./index.md#train) /
Train Option

> Auto-generated documentation for [clinicadl.train.train_option](../../../clinicadl/train/train_option.py) module.

#### Attributes

- `config_file` - train option: `click.option('--config_file', '-c', type=click.Path(exists=True), help='Path to the TOML or JSON file containing the values of the options needed for training.')`

- `gpu` - Computational: `cli_param.option_group.computational_group.option('--gpu/--no-gpu', type=bool, default=None, help='Use GPU by default. Please specify `--no-gpu` to force using CPU.')`

- `seed` - Reproducibility: `cli_param.option_group.reproducibility_group.option('--seed', help='Value to set the seed for all random operations.Default will sample a random value for the seed.', type=int)`

- `architecture` - Model: `cli_param.option_group.model_group.option('-a', '--architecture', type=str, help='Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).')`

- `label` - Task: `cli_param.option_group.task_group.option('--label', type=str, help='Target label used for training.')`

- `multi_cohort` - Data: `cli_param.option_group.data_group.option('--multi_cohort/--single_cohort', type=bool, default=None, help='Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.')`

- `n_splits` - Cross validation: `cli_param.option_group.cross_validation.option('--n_splits', type=int, help='If a value is given for k will load data of a k-fold CV. Default value (0) will load a single split.')`

- `optimizer` - Optimization: `cli_param.option_group.optimization_group.option('--optimizer', type=click.Choice(['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'NAdam', 'RAdam', 'RMSprop', 'SGD']), help='Optimizer used to train the network.')`

- `transfer_path` - transfer learning: `cli_param.option_group.transfer_learning_group.option('-tp', '--transfer_path', type=click.Path(), help='Path of to a MAPS used for transfer learning.')`
- [Train Option](#train-option)
