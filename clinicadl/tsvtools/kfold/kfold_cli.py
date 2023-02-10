import click

from clinicadl.utils import cli_param


@click.command(name="kfold", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.option.subset_name
@click.option(
    "--n_splits",
    help="Number of folds in the k-fold split. "
    "If 0, there is no training set and the whole dataset is considered as a test set.",
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
    "--merged-tsv",
    help="Path to the merged.tsv file.",
    type=str,
    default=None,
)
def cli(
    data_tsv,
    n_splits,
    subset_name,
    stratification,
    merged_tsv,
):
    """Performs a k-fold split to prepare training.

    DATA_TSV is the path to the output of tsvtool getlabels command.

    N_SPLITS is k, the number of folds of the k-fold.
    """
    from .kfold import split_diagnoses

    split_diagnoses(
        data_tsv,
        n_splits=n_splits,
        subset_name=subset_name,
        stratification=stratification,
        merged_tsv=merged_tsv,
    )


if __name__ == "__main__":
    cli()
