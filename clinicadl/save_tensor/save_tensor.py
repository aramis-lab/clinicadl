# coding: utf8
from clinicadl import MapsManager


def save_tensor(
    maps_dir,
    data_group,
    caps_directory,
    tsv_path,
    gpu=True,
    num_workers=0,
    batch_size=1,
    # prepare_dl=True,
    selection_metrics=None,
    diagnoses=None,
    multi_cohort=False,
    overwrite=False,
):
    """TO CHANGE
    This function loads a MAPS and predicts the global metrics and individual values
    for all the models selected using a metric in selection_metrics.

    Args:
        maps_dir (str): file with the model (pth format).
        data_group: prefix of all classification outputs.
        caps_directory (str): path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
        tsv_path (str): path to a TSV file containing the list of participants and sessions to interpret.
        labels: by default is True. If False no metrics tsv files will be written.
        gpu: if true, it uses gpu.
        num_workers: num_workers used in DataLoader
        batch_size: batch size of the DataLoader
        prepare_dl: if true, uses extracted patches/slices otherwise extract them
        on-the-fly.
        selection_metrics: list of metrics to find best models to be evaluated.
        diagnoses: list of diagnoses to be tested if tsv_path is a folder.
        verbose: level of verbosity.
        multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
        overwrite (bool): If True former definition of data group is erased
    """
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(maps_dir, verbose=verbose_list[0])
    maps_manager.save_tensors(
        data_group,
        caps_directory=caps_directory,
        tsv_path=tsv_path,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnoses=diagnoses,
        # prepare_dl=prepare_dl,
        # batch_size=batch_size,
        # num_workers=num_workers,
        use_cpu=not gpu,
        overwrite=overwrite,
    )
