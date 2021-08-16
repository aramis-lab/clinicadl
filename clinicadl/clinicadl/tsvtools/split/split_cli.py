import click

from clinicadl.utils import cli_param


@click.command(name="split")
@cli_param.argument.formatted_data_directory
@cli_param.option.subset_name
@cli_param.option.no_mci_sub_categories
@click.option(
    "--n_test",
    help="- If >= 1, number of subjects to put in set with name 'subset_name'.\n\n "
    "- If < 1, proportion of subjects to put set with name 'subset_name'.\n\n "
    "- If 0, no training set is created and the whole dataset is considered as one set with name 'subset_name'.",
    type=float,
    default=100.0,
    show_default=True,
)
@click.option(
    "--p_sex_threshold",
    "-ps",
    help="The threshold used for the chi2 test on sex distributions.",
    default=0.80,
    type=float,
)
@click.option(
    "--p_age_threshold",
    "-pa",
    help="The threshold used for the T-test on age distributions.",
    default=0.80,
    type=float,
)
@click.option(
    "--ignore_demographics",
    help="If given do not use age and sex to balance the split.",
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "--categorical_split_variable",
    help="Name of a categorical variable used for a stratified shuffle split "
    "(in addition to age and sex selection).",
    default=None,
    type=str,
)
def cli(
    formatted_data_directory,
    subset_name,
    n_test,
    no_mci_sub_categories,
    p_sex_threshold,
    p_age_threshold,
    ignore_demographics,
    categorical_split_variable,
):
    """Performs a single split to prepare training.

    FORMATTED_DATA_DIRECTORY is the path to the folder where the outputs of tsvtool getlabels command are stored.

    The split is done with respect to age and sex distribution.
    """
    from .split import split_diagnoses

    split_diagnoses(
        formatted_data_directory,
        n_test=n_test,
        subset_name=subset_name,
        MCI_sub_categories=no_mci_sub_categories,
        p_age_threshold=p_age_threshold,
        p_sex_threshold=p_sex_threshold,
        ignore_demographics=ignore_demographics,
        categorical_split_variable=categorical_split_variable,
    )


if __name__ == "__main__":
    cli()
