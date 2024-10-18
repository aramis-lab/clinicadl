from pathlib import Path

import click

from clinicadl.commandline import arguments
from clinicadl.commandline.modules_options import (
    computational,
    data,
    dataloader,
    maps_manager,
    validation,
)
from clinicadl.commandline.pipelines.interpret import options
from clinicadl.interpret.config import InterpretConfig
from clinicadl.predictor.predictor import Predictor


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
    from clinicadl.utils.iotools.train_utils import merge_cli_and_maps_json_options

    dict_ = merge_cli_and_maps_json_options(
        Path(kwargs["input_maps"]) / "maps.json", **kwargs
    )
    interpret_config = InterpretConfig(**dict_)
    predict_manager = Predictor(interpret_config)
    predict_manager.interpret()


if __name__ == "__main__":
    cli()
