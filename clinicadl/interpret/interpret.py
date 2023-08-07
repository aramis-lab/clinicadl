from pathlib import Path
from typing import List

from clinicadl import MapsManager


def interpret(
    maps_dir: Path,
    data_group: str,
    name: str,
    method: str,
    caps_directory: Path,
    tsv_path: Path,
    selection_metrics: List[str],
    diagnoses: List[str],
    multi_cohort: bool,
    target_node: int,
    save_individual: bool,
    batch_size: int,
    n_proc: int,
    gpu: bool,
    amp: bool,
    verbose=0,
    overwrite: bool = False,
    overwrite_name: bool = False,
    level: int = None,
    save_nifti: bool = False,
):
    """
    This function loads a MAPS and interprets all the models selected using a metric in selection_metrics.

    Parameters
    ----------
    maps_dir: str (Path)
        Path to the MAPS
    data_group: str
        Name of the data group interpreted.
    name: str
        Name of the interpretation procedure.
    method: str
        Method used for extraction (ex: gradients, grad-cam...).
    caps_directory: str (Path)
        Path to the CAPS folder. For more information please refer to
        [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
        Default will load the value of an existing data group.
    tsv_path: str (Path)
        Path to a TSV file containing the list of participants and sessions to test.
        Default will load the DataFrame of an existing data group.
    selection_metrics: list of str
        List of metrics to find best models to be evaluated..
        Default performs the interpretation on all selection metrics available.
    multi_cohort: bool
        If True caps_directory is the path to a TSV file linking cohort names and paths.
    diagnoses: list of str
        List of diagnoses to load if tsv_path is a split_directory.
        Default uses the same as in training step.
    target_node: int
        Node from which the interpretation is computed.
    save_individual: bool
        If True saves the individual map of each participant / session couple.
    batch_size: int
        If given, sets the value of batch_size, else use the same as in training step.
    n_proc: int
        If given, sets the value of num_workers, else use the same as in training step.
    gpu: bool
        If given, a new value for the device of the model will be computed.
    amp: bool
        If enabled, uses Automatic Mixed Precision (requires GPU usage).
    overwrite: bool
        If True former definition of data group is erased.
    overwrite_name: bool
        If True former interpretability map with the same name is erased.
    level: int
        Layer number in the convolutional part after which the feature map is chosen.
    save_nifi : bool
        If True, save the interpretation map in nifti format.
    verbose: int
        Level of verbosity (0: warning, 1: info, 2: debug).
    """

    verbose_list = ["warning", "info", "debug"]
    if verbose > 2:
        verbose_str = "debug"
    else:
        verbose_str = verbose_list[verbose]

    maps_manager = MapsManager(maps_dir, verbose=verbose_str)

    maps_manager.interpret(
        data_group=data_group,
        name=name,
        method=method,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        n_proc=n_proc,
        gpu=gpu,
        amp=amp,
        overwrite=overwrite,
        overwrite_name=overwrite_name,
        level=level,
        save_nifti=save_nifti,
    )
