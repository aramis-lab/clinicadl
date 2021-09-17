import click

from clinicadl.utils import cli_param


@click.command(name="save-tensor")
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.option(
    "--caps_directory",
    type=click.Path(exists=True),
    default=None,
    help="Data using CAPS structure, if different from the one used during network training.",
)
@click.option(
    "--participants_tsv",
    default=None,
    type=click.Path(),
    help="""Path to the file with subjects/sessions to process, if different from the one used during network training.
    If it includes the filename will load the TSV file directly.
    Else will load the baseline TSV files of wanted diagnoses produced by `tsvtool split`.""",
)
# @click.option(
#     "--use_extracted_features",
#     type=bool,
#     default=False,
#     is_flag=True,
#     help="""If True the extracted modes are used, otherwise they
#             will be extracted on-the-fly from the image (if mode != `image`).""",
# )
@click.option(
    "--selection_metrics",
    "-sm",
    type=click.Choice(["loss", "balanced_accuracy"]),
    default=["loss"],
    multiple=True,
    help="""Allow to select a list of models based on their selection metric. Default will
    only infer the result of the best model selected on loss.""",
)
@click.option(
    "--multi_cohort",
    type=bool,
    default=False,
    is_flag=True,
    help="""Allow to use multiple CAPS directories.
            In this case, CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files.""",
)
@click.option(
    "--diagnoses",
    "-d",
    type=click.Choice(["AD", "CN", "MCI", "sMCI", "pMCI"]),
    # default=(),
    multiple=True,
    help="List of diagnoses used for inference. Is used only if PARTICIPANTS_TSV leads to a folder.",
)
@cli_param.option.use_gpu
@cli_param.option.n_proc
@cli_param.option.batch_size
def cli(
    input_maps_directory,
    data_group,
    caps_directory,
    participants_tsv,
    gpu,
    n_proc,
    batch_size,
    # use_extracted_features,
    selection_metrics,
    diagnoses,
    multi_cohort,
):
    """Save the output tensors of a trained model on a test set.

    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model used for prediction will be loaded.

    DATA_GROUP is the name of the subjects and sessions list used to compute outputs.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()

    from .save_tensor import save_tensor

    save_tensor(
        maps_dir=input_maps_directory,
        data_group=data_group,
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        gpu=gpu,
        num_workers=n_proc,
        batch_size=batch_size,
        # prepare_dl=use_extracted_features,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
    )
