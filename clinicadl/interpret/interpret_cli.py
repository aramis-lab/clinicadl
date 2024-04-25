import click

from clinicadl.interpret import interpret_param
from clinicadl.predict.predict_config import InterpretConfig
from clinicadl.predict.predict_manager import PredictManager
from clinicadl.utils.exceptions import ClinicaDLArgumentError

config = InterpretConfig.model_fields


@click.command("interpret", no_args_is_help=True)
@interpret_param.input_maps
@interpret_param.data_group
@interpret_param.name
@interpret_param.method
@interpret_param.level
@interpret_param.selection_metrics
@interpret_param.participants_list
@interpret_param.caps_directory
@interpret_param.multi_cohort
@interpret_param.diagnoses
@interpret_param.target_node
@interpret_param.save_individual
@interpret_param.n_proc
@interpret_param.gpu
@interpret_param.amp
@interpret_param.batch_size
@interpret_param.overwrite
@interpret_param.overwrite_name
@interpret_param.save_nifti
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
