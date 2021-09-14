from typing import List

from clinicadl import MapsManager


def interpret(
    maps_dir: str,
    data_group: str,
    name: str,
    caps_directory: str,
    tsv_path: str,
    selection_metrics: List[str],
    diagnoses: List[str],
    multi_cohort: bool,
    target_node: int,
    save_individual: bool,
    batch_size: int,
    n_proc: int,
    gpu: bool,
    verbose=0,
    overwrite: bool = False,
    overwrite_name: bool = False,
):
    """
    This function loads a MAPS and interprets all the models selected using a metric in selection_metrics.

    Args:
        maps_dir: path to the MAPS.
        data_group: name of the data group interpreted.
        caps_directory: path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
        tsv_path: path to a TSV file containing the list of participants and sessions to interpret.
        target_node: Node from which the interpretation is computed.
        save_individual: If True saves the individual map of each participant / session couple.
        gpu: if true, it uses gpu.
        n_proc: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        verbose: level of verbosity (0: warning, 1: info, 2: debug).
        overwrite: If True former definition of data group is erased.
        overwrite_name: If True former interpretability map with the same name is erased.
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
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        diagnoses=diagnoses,
        multi_cohort=multi_cohort,
        target_node=target_node,
        save_individual=save_individual,
        batch_size=batch_size,
        num_workers=n_proc,
        use_cpu=not gpu,
        overwrite=overwrite,
        overwrite_name=overwrite_name,
    )
