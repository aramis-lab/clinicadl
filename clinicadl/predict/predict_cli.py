import click

from clinicadl.predict import predict_param
from clinicadl.predict.predict_config import PredictConfig
from clinicadl.predict.predict_manager import PredictManager
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.exceptions import ClinicaDLArgumentError

config = PredictConfig.model_fields


@click.command(name="predict", no_args_is_help=True)
@predict_param.input_maps
@predict_param.data_group
@predict_param.caps_directory
@predict_param.participants_list
@predict_param.use_labels
@predict_param.multi_cohort
@predict_param.diagnoses
@predict_param.label
@predict_param.save_tensor
@predict_param.save_nifti
@predict_param.save_latent_tensor
@predict_param.skip_leak_check
@predict_param.split
@predict_param.selection_metrics
@predict_param.gpu
@predict_param.amp
@predict_param.n_proc
@predict_param.batch_size
@predict_param.overwrite
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

    predict_config = PredictConfig(
        maps_dir=input_maps_directory,
        data_group=data_group,
        tsv_path=kwargs["participants_tsv"],
        split_list=kwargs["split"],
    )

    predict_manager = PredictManager(predict_config)
    predict_manager.predict()


if __name__ == "__main__":
    cli()
