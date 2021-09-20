# coding: utf8
from typing import List

from clinicadl import MapsManager


def predict(
    maps_dir: str,
    data_group: str,
    caps_directory: str,
    tsv_path: str,
    labels: bool = True,
    gpu: bool = True,
    num_workers: int = 0,
    batch_size: int = 1,
    selection_metrics: List[str] = None,
    diagnoses: List[str] = None,
    multi_cohort: bool = False,
    overwrite: bool = False,
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
        labels: by default is True. If False no metrics tsv files will be written.
        gpu: if true, it uses gpu.
        num_workers: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        multi_cohort: If True caps_directory is the path to a TSV file linking cohort names and paths.
        overwrite: If True former definition of data group is erased
    """
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_dir, verbose=verbose_list[0])
    maps_manager.predict(
        data_group,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnoses=diagnoses,
        use_labels=labels,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cpu=not gpu,
        overwrite=overwrite,
    )
