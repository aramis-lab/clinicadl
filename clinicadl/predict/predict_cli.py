from pathlib import Path

import click

from clinicadl import MapsManager
from clinicadl.predict.predict_config import PredictConfig
from clinicadl.utils import cli_param
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.predict_manager.predict_manager import PredictManager

config = PredictConfig.model_fields


@click.command(name="predict", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.option(
    "--caps_directory",
    type=config["caps_directory"].annotation,  # Path
    default=config["caps_directory"].default,  # None
    help="Data using CAPS structure, if different from the one used during network training.",
)
@click.option(
    "--participants_tsv",
    type=config["tsv_path"].annotation,  # Path
    default=config["tsv_path"].default,  # None
    help="""Path to the file with subjects/sessions to process, if different from the one used during network training.
    If it includes the filename will load the TSV file directly.
    Else will load the baseline TSV files of wanted diagnoses produced by `tsvtool split`.""",
)
@click.option(
    "--use_labels/--no_labels",
    type=config["use_labels"].annotation,  # bool
    default=config["use_labels"].default,  # false
    help="Set this option to --no_labels if your dataset does not contain ground truth labels.",
)
@click.option(
    "--selection_metrics",
    "-sm",
    type=config["selection_metrics"].annotation,  # str list ?
    default=config["selection_metrics"].default,  # ["loss"]
    multiple=True,
    help="""Allow to select a list of models based on their selection metric. Default will
    only infer the result of the best model selected on loss.""",
)
@click.option(
    "--multi_cohort",
    type=config["multi_cohort"].annotation,  # bool
    default=config["multi_cohort"].default,  # false
    is_flag=True,
    help="""Allow to use multiple CAPS directories.
            In this case, CAPS_DIRECTORY and PARTICIPANTS_TSV must be paths to TSV files.""",
)
@click.option(
    "--diagnoses",
    "-d",
    type=config["diagnoses"].annotation,  # str list ?
    default=config["diagnoses"].default,  # ??
    multiple=True,
    help="List of diagnoses used for inference. Is used only if PARTICIPANTS_TSV leads to a folder.",
)
@click.option(
    "--label",
    type=config["label"].annotation,  # str
    default=config["label"].default,  # None
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]). "
    "Default will reuse the same label as during the training task.",
)
@click.option(
    "--save_tensor",
    type=config["save_tensor"].annotation,  # bool
    default=config["save_tensor"].default,  # false
    is_flag=True,
    help="Save the reconstruction output in the MAPS in Pytorch tensor format.",
)
@cli_param.option.save_nifti
@click.option(
    "--save_latent_tensor",
    type=config["save_latent_tensor"].annotation,  # bool
    default=config["save_latent_tensor"].default,  # false
    is_flag=True,
    help="""Save the latent representation of the image.""",
)
@click.option(
    "--skip_leak_check",
    type=config["skip_leak_check"].annotation,  # bool
    default=config["skip_leak_check"].default,  # false
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
def cli(pipeline="interpret", **kwargs):
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

    if kwargs["gpu"]:
        check_gpu()
    elif kwargs["amp"]:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    predict_config = PredictConfig(
        maps_dir=kwargs["input_maps_directory"],
        data_group=kwargs["data_group"],
        caps_directory=kwargs["caps_directory"],
        tsv_path=kwargs["tsv_path"],
        use_labels=kwargs["use_labels"],
        label=kwargs["label"],
        gpu=kwargs["gpu"],
        amp=kwargs["amp"],
        n_proc=kwargs["n_proc"],
        batch_size=kwargs["batch_size"],
        split_list=kwargs["split"],
        selection_metrics=kwargs["selection_metrics"],
        diagnoses=kwargs["diagnoses"],
        multi_cohort=kwargs["multi_cohort"],
        overwrite=kwargs["overwrite"],
        save_tensor=kwargs["save_tensor"],
        save_nifti=kwargs["save_nifti"],
        save_latent_tensor=kwargs["save_latent_tensor"],
        skip_leak_check=kwargs["skip_leak_check"],
    )

    predict_manager = PredictManager(predict_config)
    predict_manager.predict()


if __name__ == "__main__":
    cli()
