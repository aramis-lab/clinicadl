from pathlib import Path

import click

from clinicadl import MapsManager
from clinicadl.predict.predict_config import PredictConfig
from clinicadl.utils import cli_param
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.predict_manager.predict_manager import PredictManager


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
    """This function loads a MAPS and predicts the global metrics and individual values
    for all the models selected using a metric in selection_metrics.

    Args:
        maps_dir: path to the MAPS.
        data_group: name of the data group tested.
        caps_directory: path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
        tsv_path: path to a TSV file containing the list of participants and sessions to interpret.
        use_labels: by default is True. If False no metrics tsv files will be written.
        label: Name of the target value, if different from training.
        gpu: if true, it uses gpu.
        amp: If enabled, uses Automatic Mixed Precision (requires GPU usage).
        n_proc: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        overwrite: If True former definition of data group is erased
        save_tensor: For reconstruction task only, if True it will save the reconstruction as .pt file in the MAPS.
        save_nifti: For reconstruction task only, if True it will save the reconstruction as NIfTI file in the MAPS.

    Infer the outputs of a trained model on a test set.

    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model used for prediction will be loaded.

    DATA_GROUP is the name of the subjects and sessions list used for the interpretation.
    """

    if gpu:
        check_gpu()
    elif amp:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    predict_config = PredictConfig(
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

    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(predict_config.maps_dir, verbose=verbose_list[0])
    predict_manager = PredictManager(maps_manager)

    # Check if task is reconstruction for "save_tensor" and "save_nifti"
    if (
        predict_config.save_tensor
        and predict_manager.maps_manager.network_task != "reconstruction"
    ):
        raise ClinicaDLArgumentError(
            "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
        )
    if (
        predict_config.save_nifti
        and predict_manager.maps_manager.network_task != "reconstruction"
    ):
        raise ClinicaDLArgumentError(
            "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
        )

    predict_manager.predict(predict_config)
