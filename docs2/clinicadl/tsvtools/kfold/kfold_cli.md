# Kfold Cli

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Kfold](./index.md#kfold) /
Kfold Cli

> Auto-generated documentation for [clinicadl.tsvtools.kfold.kfold_cli](../../../../clinicadl/tsvtools/kfold/kfold_cli.py) module.

- [Kfold Cli](#kfold-cli)
  - [cli](#cli)

## cli

[Show source in kfold_cli.py:6](../../../../clinicadl/tsvtools/kfold/kfold_cli.py#L6)

Performs a k-fold split to prepare training.

DATA_TSV is the path to the output of tsvtool getlabels command.

N_SPLITS is k, the number of folds of the k-fold.

#### Signature

```python
@click.command(name="kfold", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.option.subset_name
@click.option(
    "--n_splits",
    help=(
        "Number of folds in the k-fold split. If 0, there is no training set and the"
        " whole dataset is considered as a test set."
    ),
    show_default=True,
    type=int,
    default=5,
)
@click.option(
    "--stratification",
    help="Name of a variable used to stratify the k-fold split.",
    type=str,
    default=None,
)
@click.option(
    "--merged-tsv", help="Path to the merged.tsv file.", type=str, default=None
)
def cli(data_tsv, n_splits, subset_name, stratification, merged_tsv):
    ...
```