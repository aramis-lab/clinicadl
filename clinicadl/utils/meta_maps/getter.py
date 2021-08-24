"""
Produces a tsv file to analyze the performance of one launch of the random search.
"""
import os
from os import path

import pandas as pd

from clinicadl import MapsManager


def meta_maps_analysis(launch_dir, evaluation_metric="loss"):
    """
    This function summarizes the validation performance according to `evaluation_metric`
    of several MAPS stored in the folder `launch_dir`.
    The output TSV files are written in `launch_dir`.

    Args:
        launch_dir (str): Path to the directory containing several MAPS.
        evaluation_metric (str): Name of the metric used for validation evaluation.
    """

    jobs_list = [
        job
        for job in os.listdir(launch_dir)
        if path.exists(path.join(launch_dir, job, "maps.json"))
    ]

    selection_set = set()  # Set of all selection metrics seen
    folds_set = set()  # Set of all folds seen

    performances_dict = dict()
    for job in jobs_list:
        performances_dict[job] = dict()
        maps_manager = MapsManager(path.join(launch_dir, job))
        folds = maps_manager._find_folds()
        folds_set = folds_set | set(folds)
        for fold in folds:
            performances_dict[job][fold] = dict()
            selection_metrics = maps_manager._find_selection_metrics(fold)
            selection_set = selection_set | set(selection_metrics)
            for metric in selection_metrics:
                validation_metrics = maps_manager.get_metrics(
                    "validation", fold, metric
                )
                if evaluation_metric not in validation_metrics:
                    raise ValueError(
                        f"Evaluation metric {evaluation_metric} not found in "
                        f"MAPS {job}, for fold {fold} and selection {metric}."
                    )
                performances_dict[job][fold][metric] = validation_metrics[
                    evaluation_metric
                ]

    # Produce one analysis for each selection metric
    for metric in selection_set:
        df = pd.DataFrame()
        filename = f"analysis_metric-{evaluation_metric}_selection-{metric}.tsv"
        for job in jobs_list:
            for fold in folds_set:
                df.loc[job, f"fold-{fold}"] = performances_dict[job][fold][metric]
        df.to_csv(path.join(launch_dir, filename), sep="\t")
