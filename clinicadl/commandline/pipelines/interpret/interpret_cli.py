import click

from clinicadl.commandline import arguments
from clinicadl.commandline.pipelines.interpret import options
from clinicadl.config.config.pipelines.interpret import InterpretConfig
from clinicadl.config.options import (
    computational,
    data,
    dataloader,
    maps_manager,
    validation,
)
from clinicadl.predict.predict_manager import PredictManager


@click.command("interpret", no_args_is_help=True)
@arguments.input_maps
@arguments.data_group
@maps_manager.overwrite
@maps_manager.save_nifti
@options.name
@options.method
@options.level
@options.target_node
@options.save_individual
@options.overwrite_name
@data.participants_tsv
@data.caps_directory
@data.multi_cohort
@data.diagnoses
@dataloader.n_proc
@dataloader.batch_size
@computational.gpu
@computational.amp
@validation.selection_metrics
def cli(**kwargs):
    """Interpretation of trained models using saliency map method.
    INPUT_MAPS_DIRECTORY is the MAPS folder from where the model to interpret will be loaded.
    DATA_GROUP is the name of the subjects and sessions list used for the interpretation.
    NAME is the name of the saliency map task.
    METHOD is the method used to extract an attribution map.
    """

    interpret_config = InterpretConfig(**kwargs)
    predict_manager = PredictManager(interpret_config)
    predict_manager.interpret()


if __name__ == "__main__":
    cli()
