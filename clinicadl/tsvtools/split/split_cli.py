import click

from clinicadl.utils import cli_param


@click.command(name="split", no_args_is_help=True)
@cli_param.argument.data_tsv
@cli_param.option.subset_name
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
    "(in addition to age, sex and group selection).",
    default=None,
    type=str,
)
@click.option(
    "--not_only_keep_baseline",
    help="If given will store the file with all subjects",
    default=False,
    is_flag=True,
    type=bool,
)
@click.option(
    "--multi-diagnoses",
    help="If given, all columns are used to balance the split, not only age and sex",
    default=False,
    is_flag=True,
    type=bool,
)
def cli(
    data_tsv,
    subset_name,
    n_test,
    p_sex_threshold,
    p_age_threshold,
    ignore_demographics,
    categorical_split_variable,
    not_only_keep_baseline,
    multi_diagnoses,
):
    """Performs a single split to prepare training.

    DATA_TSV is the path to the tsv file where the outputs of tsvtool getlabels command are stored.

    The split is done with respect to age, sex and group distribution.
    """
    from .split import split_diagnoses

    split_diagnoses(
        data_tsv,
        n_test=n_test,
        subset_name=subset_name,
        p_age_threshold=p_age_threshold,
        p_sex_threshold=p_sex_threshold,
        ignore_demographics=ignore_demographics,
        categorical_split_variable=categorical_split_variable,
        not_only_baseline=not_only_keep_baseline,
        multi_diagnoses=multi_diagnoses,
    )


if __name__ == "__main__":
    cli()
