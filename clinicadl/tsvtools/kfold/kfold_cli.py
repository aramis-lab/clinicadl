import click

from clinicadl.utils import cli_param


@click.command(name="kfold", no_args_is_help=True)
@cli_param.argument.formatted_data_tsv
@cli_param.option.subset_name
@click.option(
    "--test_tsv",
    "-tt",
    help="Name of the test file in tsv format",
    type=str,
    default=None,
)
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
def cli(
    formatted_data_tsv,
    n_splits,
    subset_name,
    stratification,
    test_tsv,
):
    """Performs a k-fold split to prepare training.

    FORMATTED_DATA_DIRECTORY is the path to the folder where the outputs of tsvtool getlabels command are stored.

    N_SPLITS is k, the number of folds of the k-fold.
    """
    from .kfold import split_diagnoses

    split_diagnoses(
        formatted_data_tsv,
        n_splits=n_splits,
        subset_name=subset_name,
        stratification=stratification,
        test_tsv=test_tsv,
    )


if __name__ == "__main__":
    cli()
