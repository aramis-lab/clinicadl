import click
import pandas as pd

from clinicadl.utils import cli_param


@click.command(name="prepare-experiment", no_args_is_help=True)
@cli_param.argument.data_tsv
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
    data_tsv,
    n_test,
    validation_type,
    n_validation,
):
    """Performs a single split to prepare testing data and then can perform either k-fold or single split to prepare validation data.

    DATA_TSV is the path to the folder where the outputs of tsvtool getlabels command are stored.

    N_TEST is the numbers/proportion of subjects to put in the test split.

    VALIDATION_TYPE is the type of split wanted for the validation split.

    N_VALIDATION is the numbers/proportion of subjects to put in the validation split.


    The split is done with respect to age and sex distribution.
    Threshold on the p-value used for the T-test on age distributions is 0.80.
    Threshold on the p-value used for the chi2 test on sex distributions is 0.80.
    No variable are used to stratify the k-fold split.
    """

    from clinicadl.tsvtools.split import split_diagnoses

    p_age_threshold = 0.80
    p_sex_threshold = 0.80
    ignore_demographics = False
    flag_not_baseline = False
    split_diagnoses(
        data_tsv,
        n_test=n_test,
        subset_name="test",
        p_age_threshold=p_age_threshold,
        p_sex_threshold=p_sex_threshold,
        ignore_demographics=ignore_demographics,
        categorical_split_variable=None,
        not_only_baseline=flag_not_baseline,
    )

    parents_path = data_tsv.parents[0]
    split_numero = 1
    folder_name = "split"

    while (parents_path / folder_name).is_dir():
        split_numero += 1
        folder_name = f"split_{split_numero}"
    if split_numero > 2:
        folder_name = f"split_{split_numero-1}"
    else:
        folder_name = "split"

    results_path = parents_path / folder_name
    train_tsv = results_path / "train.tsv"

    train_df = pd.read_csv(train_tsv, sep="\t")
    list_columns = train_df.columns.values
    if (
        "diagnosis" not in list_columns
        or ("age" not in list_columns and "age_bl" not in list_columns)
        or "sex" not in list_columns
    ):
        data_df = pd.read_csv(data_tsv, sep="\t")
        train_df = pd.merge(
            train_df,
            data_df,
            how="inner",
            on=["participant_id", "session_id"],
        )
        train_df.to_csv(train_tsv, sep="\t")

    if validation_type == "split":
        split_diagnoses(
            train_tsv,
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
            train_tsv,
            n_splits=int(n_validation),
            subset_name="validation",
            stratification=None,
        )


if __name__ == "__main__":
    cli()
