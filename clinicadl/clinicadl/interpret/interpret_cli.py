import click

from clinicadl.utils import cli_param


@click.command("interpret")
@cli_param.argument.input_maps
@click.argument(
    "interpretation_method",
    type=str,
)
# Model
@click.option(
    "--selection_metrics",
    default=["loss"],
    type=str,
    multiple=True,
    help="Loads the model selected on the metrics given.",
)
# Data
@click.option(
    "--participants_tsv",
    type=click.File(),
    default=None,
    help="TSV path with subjects/sessions to process, if different from classification task.",
)
@click.option(
    "--caps_directory",
    type=click.Path(exists=True),
    default=None,
    help="Data using CAPS structure, if different from classification task",
)
@click.option(
    "--multi_cohort",
    type=bool,
    default=False,
    is_flag=True,
    help="Performs multi-cohort interpretation. In this case, caps_directory and tsv_path must be paths to TSV files.",
)
@click.option(
    "-d",
    "--diagnosis",
    default="AD",
    type=str,
    help="The images corresponding to this diagnosis only will be loaded.",
)
@click.option(
    "--target_node",
    default=0,
    type=str,
    help="Which target node the gradients explain. Default takes the first output node.",
)
@click.option(
    "--baseline",
    type=bool,
    default=False,
    is_flag=True,
    help="If provided, only the baseline sessions are used for interpretation.",
)
@click.option(
    "--save_individual",
    type=str,
    default=None,
    help="Saves individual saliency maps in addition to the mean saliency map.",
)
@cli_param.option.n_proc
@cli_param.option.use_gpu
@cli_param.option.batch_size
def cli(
    input_maps_directory,
    interpretation_method,
    caps_directory,
    participants_tsv,
    selection_metrics,
    multi_cohort,
    diagnosis,
    baseline,
    target_node,
    save_individual,
    batch_size,
    n_proc,
    use_gpu,
    # verbose,
):
    """
    Interpret the prediction of INPUT_MAPS_DIRECTORY with the chosen INTERPRETATION_METHOD.
    """
    from .interpret import interpret

    interpret(
        model_path=input_maps_directory,
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        name=interpretation_method,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnosis=diagnosis,
        baseline=baseline,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        nproc=n_proc,
        use_cpu=not use_gpu,
        # verbose=verbose,
    )
