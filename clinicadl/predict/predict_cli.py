import click

from clinicadl.config import arguments
from clinicadl.config.options import (
    computational,
    cross_validation,
    data,
    dataloader,
    maps_manager,
    predict,
    validation,
)
from clinicadl.predict.pipeline_config import PredictPipelineConfig
from clinicadl.predict.predict_manager import PredictManager
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.exceptions import ClinicaDLArgumentError


@click.command(name="predict", no_args_is_help=True)
@arguments.input_maps
@arguments.data_group
@maps_manager.save_nifti
@maps_manager.overwrite
@predict.use_labels
@data.label
@predict.save_tensor
@predict.save_latent_tensor
@data.caps_directory
@data.participants_tsv
@data.multi_cohort
@data.diagnoses
@validation.skip_leak_check
@validation.selection_metrics
@cross_validation.split
@computational.gpu
@computational.amp
@dataloader.n_proc
@dataloader.batch_size
def cli(input_maps_directory, data_group, **kwargs):
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

    predict_config = PredictPipelineConfig(**kwargs)
    predict_manager = PredictManager(predict_config)
    predict_manager.predict()


if __name__ == "__main__":
    cli()
