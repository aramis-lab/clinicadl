# coding: utf8
from pathlib import Path
from typing import List

from clinicadl import MapsManager
from clinicadl.utils.exceptions import ClinicaDLArgumentError


def predict(
    maps_dir: Path,
    data_group: str,
    caps_directory: Path,
    tsv_path: Path,
    use_labels: bool = True,
    label: str = None,
    gpu: bool = True,
    amp: bool = False,
    n_proc: int = 0,
    batch_size: int = 8,
    split_list: List[int] = None,
    selection_metrics: List[str] = None,
    diagnoses: List[str] = None,
    multi_cohort: bool = False,
    overwrite: bool = False,
    save_tensor: bool = False,
    save_nifti: bool = False,
    save_latent_tensor: bool = False,
    skip_leak_check: bool = False,
):
    """
    This function loads a MAPS and predicts the global metrics and individual values
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
    """
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_dir, verbose=verbose_list[0])
    # Check if task is reconstruction for "save_tensor" and "save_nifti"
    if save_tensor and maps_manager.network_task != "reconstruction":
        raise ClinicaDLArgumentError(
            "Cannot save tensors if the network task is not reconstruction. Please remove --save_tensor option."
        )
    if save_nifti and maps_manager.network_task != "reconstruction":
        raise ClinicaDLArgumentError(
            "Cannot save nifti if the network task is not reconstruction. Please remove --save_nifti option."
        )
    maps_manager.predict(
        data_group,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        split_list=split_list,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnoses=diagnoses,
        label=label,
        use_labels=use_labels,
        batch_size=batch_size,
        n_proc=n_proc,
        gpu=gpu,
        amp=amp,
        overwrite=overwrite,
        save_tensor=save_tensor,
        save_nifti=save_nifti,
        save_latent_tensor=save_latent_tensor,
        skip_leak_check=skip_leak_check,
    )
