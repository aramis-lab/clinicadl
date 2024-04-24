from pathlib import Path
from typing import get_args

import click

from clinicadl import MapsManager
from clinicadl.predict.predict_config import InterpretConfig
from clinicadl.utils import cli_param
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.predict_manager.predict_manager import PredictManager

config = InterpretConfig.model_fields


@click.command("interpret", no_args_is_help=True)
@cli_param.argument.input_maps
@cli_param.argument.data_group
@click.argument(
    "name",
    type=config["name"].annotation,
)
@click.argument(
    "method",
    type=config["method_cls"].annotation,  # ["gradients", "grad-cam"]
)
@click.option(
    "--level_grad_cam",
    type=click.IntRange(min=1),
    default=None,
    help="level of the feature map (after the layer corresponding to the number) chosen for grad-cam.",
)
# Model
@cli_param.option.selection_metrics
# Data
@cli_param.option.participant_list
@cli_param.option.caps_directory
@click.option(
    "--multi_cohort",
    type=config["multi_cohort"].annotation,  # bool
    default=config["multi_cohort"].default,  # false
    is_flag=True,
    help="Performs multi-cohort interpretation. In this case, caps_directory and tsv_path must be paths to TSV files.",
)
@cli_param.option.diagnoses
@click.option(
    "--target_node",
    type=config["target_node"].annotation,  # int
    default=config["target_node"].default,  # 0
    help="Which target node the gradients explain. Default takes the first output node.",
)
@click.option(
    "--save_individual",
    type=config["save_individual"].annotation,  # bool
    default=config["save_individual"].default,  # false
    is_flag=True,
    help="Save individual saliency maps in addition to the mean saliency map.",
)
@cli_param.option.n_proc
@cli_param.option.use_gpu
@cli_param.option.amp
@cli_param.option.batch_size
@cli_param.option.overwrite
@click.option(
    "--overwrite_name",
    "-on",
    is_flag=True,
    type=config["overwrite_name"].annotation,  # bool
    default=config["overwrite_name"].default,  # false
    help="Overwrite the name if it already exists.",
)
@cli_param.option.save_nifti
def cli(input_maps_directory, data_group, name, method, **kwargs):
    """Interpretation of trained models using saliency map method.

    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model to interpret will be loaded.

    DATA_GROUP is the name of the subjects and sessions list used for the interpretation.

    NAME is the name of the saliency map task.

    METHOD is the method used to extract an attribution map.
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    if kwargs["gpu"]:
        check_gpu()
    elif kwargs["amp"]:
        raise ClinicaDLArgumentError(
            "AMP is designed to work with modern GPUs. Please add the --gpu flag."
        )

    interpret_config = InterpretConfig(
        maps_dir=input_maps_directory,
        data_group=data_group,
        name=name,
        method_cls=method,
        tsv_path=kwargs["participants_tsv"],
        level=kwargs["level_grad_cam"],
        **kwargs,
    )

    predict_manager = PredictManager(interpret_config)
    predict_manager.interpret()


if __name__ == "__main__":
    cli()
