# coding: utf8
from clinicadl import MapsManager


def predict(
    caps_directory,
    tsv_path,
    model_path,
    prefix_output,
    labels=True,
    gpu=True,
    num_workers=0,
    batch_size=1,
    prepare_dl=True,
    selection_metrics=None,
    diagnoses=None,
    verbose=0,
    multi_cohort=False,
):
    """
    This function loads a MAPS and predicts the global metrics and individual values
    for all the models selected using a metric in selection_metrics.

    Args:
        caps_directory (str): path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
        tsv_path (str): path to a TSV file containing the list of participants and sessions to interpret.
        model_path (str): file with the model (pth format).
        prefix_output: prefix of all classification outputs.
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

    """
    verbose_list = ["warning", "info", "debug"]

    maps_manager = MapsManager(model_path, verbose=verbose_list[verbose])
    maps_manager.predict(
        caps_directory,
        tsv_path,
        prefix_output,
        selection_metrics=selection_metrics,
        multi_cohort=multi_cohort,
        diagnoses=diagnoses,
        use_labels=labels,
        prepare_dl=prepare_dl,
        batch_size=batch_size,
        num_workers=num_workers,
        use_cpu=not gpu,
    )
