from pathlib import Path

import click

from clinicadl.utils import cli_param
from clinicadl.utils.exceptions import ClinicaDLArgumentError


@click.command(name="predict", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.option(
    "--caps_directory",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Data using CAPS structure, if different from the one used during network training.",
)
@click.option(
    "--participants_tsv",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="""Path to the file with subjects/sessions to process, if different from the one used during network training.
    If it includes the filename will load the TSV file directly.
    Else will load the baseline TSV files of wanted diagnoses produced by `tsvtool split`.""",
)
@click.option(
    "--use_labels/--no_labels",
    default=True,
    help="Set this option to --no_labels if your dataset does not contain ground truth labels.",
)
@click.option(
    "--selection_metrics",
    "-sm",
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
    type=str,
    multiple=True,
    help="List of diagnoses used for inference. Is used only if PARTICIPANTS_TSV leads to a folder.",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
@click.option(
    "--save_tensor",
    type=bool,
    default=False,
    is_flag=True,
    help="Save the reconstruction output in the MAPS in Pytorch tensor format.",
)
@cli_param.option.save_nifti
@click.option(
    "--save_latent_tensor",
    type=bool,
    default=False,
    is_flag=True,
    help="""Save the latent representation of the image.""",
)
@click.option(
    "--skip_leak_check",
    type=bool,
    default=False,
    is_flag=True,
    help="Skip the data leakage check.",
)
@cli_param.option.split
@cli_param.option.selection_metrics
@cli_param.option.use_gpu
@cli_param.option.amp
@cli_param.option.n_proc
@cli_param.option.batch_size
@cli_param.option.overwrite
def cli(
    input_maps_directory,
    data_group,
    caps_directory,
    participants_tsv,
    split,
    gpu,
    amp,
    n_proc,
    batch_size,
    use_labels,
    label,
    selection_metrics,
    diagnoses,
    multi_cohort,
    overwrite,
    save_tensor,
    save_nifti,
    save_latent_tensor,
    skip_leak_check,
):
    """Infer the outputs of a trained model on a test set.

    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model used for prediction will be loaded.

    DATA_GROUP is the name of the subjects and sessions list used for the interpretation.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if gpu:
        check_gpu()
    elif amp:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    from .predict import predict

    predict(
        maps_dir=input_maps_directory,
        data_group=data_group,
        caps_directory=caps_directory,
        tsv_path=participants_tsv,
        use_labels=use_labels,
        label=label,
        gpu=gpu,
        amp=amp,
        n_proc=n_proc,
        batch_size=batch_size,
        split_list=split,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
        overwrite=overwrite,
        save_tensor=save_tensor,
        save_nifti=save_nifti,
        save_latent_tensor=save_latent_tensor,
        skip_leak_check=skip_leak_check,
    )
