import click

from clinicadl.utils import cli_param


@click.command(name="predict")
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.option(
    "--caps_directory",
    type=click.Path(exists=True),
    default=None,
    help="Data using CAPS structure, if different from classification task",
)
@click.option(
    "--participants_tsv",
    default=None,
    type=click.Path(),
    help="""Path to the file with subjects/sessions to process.
    If it includes the filename will load the tsv file directly.
    Else will load the baseline tsv files of wanted diagnoses produced by tsvtool.""",
)
@click.option(
    "--labels/--no_labels",
    default=False,
    help="Set this to --label if your dataset does not contain a ground truth.",
)
@click.option(
    "--use_extracted_features",
    type=bool,
    default=False,
    is_flag=True,
    help="""If True the extract slices or patche are used, otherwise the they
            will be extracted on the fly (if necessary).""",
)
@click.option(
    "--selection_metrics",
    "-sm",
    type=click.Choice(["loss", "balanced_accuracy"]),
    default=["balanced_accuracy"],
    multiple=True,
    help="""List of metrics to find the best models to evaluate. Default will
    classify best model based on balanced accuracy.""",
)
@click.option(
    "--multi_cohort",
    type=bool,
    default=False,
    is_flag=True,
    help="""Performs multi-cohort classification.
            In this case, CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files.""",
)
@click.option(
    "--diagnoses",
    "-d",
    type=click.Choice(["AD", "CN", "MCI", "sMCI", "pMCI"]),
    # default=(),
    multiple=True,
    help="List of participants diagnoses that will be classified.",
)
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
def cli(
    input_maps,
    data_group,
    caps_directory,
    participants_tsv,
    gpu,
    n_proc,
    batch_size,
    labels,
    use_extracted_features,
    selection_metrics,
    diagnoses,
    multi_cohort,
):
    """
    Compute prediction of DATA_GROUP data with INPUT_MAPS_DIRECTORY models.
    """
    from .predict import predict

    predict(
        maps_dir=input_maps,
        data_group=data_group,
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        labels=labels,
        gpu=gpu,
        num_workers=n_proc,
        batch_size=batch_size,
        prepare_dl=use_extracted_features,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
    )
