import click

from clinicadl.config import arguments
from clinicadl.config.options import (
    computational,
    data,
    dataloader,
    interpret,
    maps_manager,
    validation,
)
from clinicadl.interpret.pipeline_config import InterpretPipelineConfig
from clinicadl.predict.predict_manager import PredictManager
from clinicadl.utils.exceptions import ClinicaDLArgumentError


@click.command("interpret", no_args_is_help=True)
@maps_manager.maps_dir
@maps_manager.data_group
@maps_manager.overwrite
@maps_manager.save_nifti
@interpret.name
@interpret.method
@interpret.level
@interpret.target_node
@interpret.save_individual
@interpret.overwrite_name
@data.participants_tsv
@data.caps_directory
@data.multi_cohort
@data.diagnoses
@dataloader.n_proc
@dataloader.batch_size
@computational.gpu
@computational.amp
@validation.selection_metrics
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

    interpret_config = InterpretPipelineConfig(**kwargs)

    predict_manager = PredictManager(interpret_config)
    predict_manager.interpret()


if __name__ == "__main__":
    cli()
