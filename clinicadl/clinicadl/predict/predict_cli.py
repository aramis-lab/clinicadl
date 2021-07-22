import click

from clinicadl.utils import cli_param

cmd_name = "classify"

@click.command(name=cmd_name)
@cli_param.argument.caps_directory
@cli_param.argument.input_maps
@cli_param.argument.inference_prefix
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
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
    type=bool, default=False, is_flag=True,
    help="""If True the extract slices or patche are used, otherwise the they
            will be extracted on the fly (if necessary).""",
)
@click.option(
    "--selection_metrics", "-sm",
    type=click.Choice(["loss", "balanced_accuracy"]),
    default=("balanced_accuracy"),
    multiple=True,
    help="""List of metrics to find the best models to evaluate. Default will
    classify best model based on balanced accuracy.""",
)
@click.option(
    "--multi_cohort",
    type=bool, default=False, is_flag=True,
    help="""Performs multi-cohort classification. 
            In this case, CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files.""",
)
@click.option(
    "--diagnoses", "-d",
    type=click.Choice(["AD", "CN", "MCI", "sMCI", "pMCI"]),
    default=None,
    multiple=True,
    help="List of participants that will be classified.",
)
def cli(caps_directory, participants_tsv, input_maps, inference_prefix,
        use_gpu, n_proc, batch_size, labels, use_extracted_features,
        selection_metrics, diagnoses, multi_cohort):
    """
    """
    from .infer import classify
    classify(
        caps_dir=caps_directory,
        tsv_path=participants_tsv,
        model_path=input_maps,
        prefix_output=inference_prefix,
        labels=labels,
        gpu=use_gpu,
        num_workers=n_proc,
        batch_size=batch_size,
        prepare_dl=use_extracted_features,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
    )