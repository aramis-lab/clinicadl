import click

from clinicadl.utils import cli_param


@click.command(name="prepare-experiment", no_args_is_help=True)
@cli_param.argument.formatted_data_directory
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
    "--validation_type",
    "-vt",
    help="Type of split wanted for the validation: split or kfold",
    default="kfold",
    type=click.Choice(["split", "kfold"]),
)
@click.option(
    "--n_validation",
    help="- If it is a SingleSplit: number of subjects top put in validation set if it is a SingleSplit.\n\n"
    "- If it is a k-fold split: number of folds in the k-folds split.\n\n"
    "- If 0, there is no training set and the whole dataset is considered as a test set.",
    default=5.0,
    type=float,
)
def cli(
    formatted_data_directory,
    n_test,
    validation_type,
    n_validation,
):
    """Performs a single split to prepare testing data and then can perform either k-fold or single split to prepare validation data.

    FORMATTED_DATA_DIRECTORY is the path to the folder where the outputs of tsvtool getlabels command are stored.

    The split is done with respect to age and sex distribution.
    Threshold on the p-value used for the T-test on age distributions is 0.80.
    Threshold on the p-value used for the chi2 test on sex distributions is 0.80.
    No variable are used to stratify the k-fold split.
    """

    from clinicadl.tsvtools.split import split_diagnoses

    p_age_threshold = 0.80
    p_sex_threshold = 0.80
    ignore_demographics = False

    split_diagnoses(
        formatted_data_directory,
        n_test=n_test,
        subset_name="test",
        p_age_threshold=p_age_threshold,
        p_sex_threshold=p_sex_threshold,
        ignore_demographics=ignore_demographics,
        categorical_split_variable=None,
    )
    if validation_type == "split":
        split_diagnoses(
            formatted_data_directory,
            n_test=n_validation,
            subset_name="validation",
            p_age_threshold=p_age_threshold,
            p_sex_threshold=p_sex_threshold,
            ignore_demographics=ignore_demographics,
            categorical_split_variable=None,
        )
    elif validation_type == "kfold":
        from clinicadl.tsvtools.kfold import split_diagnoses as kfold_diagnoses

        kfold_diagnoses(
            formatted_data_directory,
            n_splits=n_validation,
            subset_name="validation",
            stratification=None,
        )


if __name__ == "__main__":
    cli()
