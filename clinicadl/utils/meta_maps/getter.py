"""
Produces a tsv file to analyze the performance of one launch of the random search.
"""

from pathlib import Path

import pandas as pd

from clinicadl import MapsManager
from clinicadl.utils.exceptions import MAPSError


def meta_maps_analysis(launch_dir: Path, evaluation_metric="loss"):
    """
    This function summarizes the validation performance according to `evaluation_metric`
    of several MAPS stored in the folder `launch_dir`.
    The output TSV files are written in `launch_dir`.

    Args:
        launch_dir (str): Path to the directory containing several MAPS.
        evaluation_metric (str): Name of the metric used for validation evaluation.
    """
    launch_dir = launch_dir
    jobs_list = [
        job
        for job in list(launch_dir.iter_dir())
        if (launch_dir / job / "maps.json").is_file()
    ]

    selection_set = set()  # Set of all selection metrics seen
    split_set = set()  # Set of all splits seen

    performances_dict = dict()
    for job in jobs_list:
        performances_dict[job] = dict()
        maps_manager = MapsManager(launch_dir / job)
        split_list = maps_manager._find_splits()
        split_set = split_set | set(split_list)
        for split in split_set:
            performances_dict[job][split] = dict()
            selection_metrics = maps_manager._find_selection_metrics(split)
            selection_set = selection_set | set(selection_metrics)
            for metric in selection_metrics:
                validation_metrics = maps_manager.get_metrics(
                    "validation", split, metric
                )
                if evaluation_metric not in validation_metrics:
                    raise MAPSError(
                        f"Evaluation metric {evaluation_metric} not found in "
                        f"MAPS {job}, for split {split} and selection {metric}."
                    )
                performances_dict[job][split][metric] = validation_metrics[
                    evaluation_metric
                ]

    # Produce one analysis for each selection metric
    for metric in selection_set:
        df = pd.DataFrame()
        filename = f"analysis_metric-{evaluation_metric}_selection-{metric}.tsv"
        for job in jobs_list:
            for split in split_set:
                df.loc[job, f"split-{split}"] = performances_dict[job][split][metric]
        df.to_csv(launch_dir / filename, sep="\t")
