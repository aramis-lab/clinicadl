import click

from clinicadl.utils import cli_param

cmd_name = "kfold"


@click.command(name=cmd_name)
@cli_param.argument.formatted_data_directory
@cli_param.option.no_MCI_sub_categories
@cli_param.option.subset_name
@click.option(
    "--n_splits",
    help="Number of folds in the k-fold split."
    "If 0, there is no training set and the whole dataset is considered as a test set.",
    type=int,
    default=5,
)
@click.option(
    "--stratification",
    help="Name of a variable used to stratify the k-fold split.",
    type=str,
    default=None,
)

def cli(formatted_data_directory, n_splits, no_MCI_sub_categories, subset_name, stratification):
    """
    """
    # import function to execute
    from .kfold import split_diagnoses
    # run function
    split_diagnoses(formatted_data_directory, n_splits=n_splits,
        subset_name=subset_name, MCI_sub_categories=no_MCI_sub_categories,
        stratification=stratification)

if __name__ == "__main__":
    cli()
