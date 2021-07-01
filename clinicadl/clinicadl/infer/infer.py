# coding: utf8
from clinicadl import MapsManager


def classify(
    caps_dir,
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
    This function verifies the input folders, and the existence of the json file
    then it launch the inference stage from a specific model.

    It writes the outputs in <model_path>/fold-<fold>/cnn_classification.

    Args:
        caps_dir: folder containing the tensor files (.pt version of MRI)
        tsv_path: file with the name of the MRIs to process (single or multiple)
        model_path: file with the model (pth format).
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
        caps_dir,
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
