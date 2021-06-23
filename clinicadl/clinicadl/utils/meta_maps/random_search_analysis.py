"""
Produces a tsv file to analyze the performance of one launch of the random search.
"""
import os
from os import path
from warnings import warn

import numpy as np
import pandas as pd

from clinicadl.utils.maps_manager import read_json


def random_search_analysis(launch_dir):

    rs_options = read_json(json_path=path.join(launch_dir, "random_search.json"))

    if rs_options.split is None:
        if rs_options.n_splits is None:
            fold_iterator = range(1)
        else:
            fold_iterator = range(rs_options.n_splits)
    else:
        fold_iterator = rs_options.split

    jobs_list = [
        job
        for job in os.listdir(launch_dir)
        if path.exists(path.join(launch_dir, job, "commandline.json"))
    ]

    for selection in ["balanced_accuracy", "loss"]:

        columns = [
            "run",
            ">0.5",
            ">0.55",
            ">0.6",
            ">0.65",
            ">0.7",
            ">0.75",
            ">0.8",
            ">0.85",
            ">0.9",
            ">0.95",
            "folds",
        ]
        output_df = pd.DataFrame(columns=columns)
        thresholds = np.arange(0.5, 1, 0.05)
        thresholds = np.insert(thresholds, 0, 0)

        for job in jobs_list:

            valid_accuracies = []
            for fold in fold_iterator:
                performance_path = path.join(
                    launch_dir,
                    job,
                    f"fold-{fold}",
                    "cnn_classification",
                    f"best_{selection}",
                )
                if path.exists(performance_path):
                    valid_df = pd.read_csv(
                        path.join(
                            performance_path, "validation_image_level_metrics.tsv"
                        ),
                        sep="\t",
                    )
                    valid_accuracies.append(
                        valid_df.loc[0, "balanced_accuracy"].astype(float)
                    )
                else:
                    warn(f"The fold {fold} doesn't exist for job {job}")

            # Find the mean value of all existing folds
            if len(valid_accuracies) > 0:
                bac_valid = np.mean(valid_accuracies)
                row = (bac_valid > thresholds).astype(int)
            else:
                row = np.zeros(len(thresholds), dtype=int)
            row = np.concatenate([row, [len(valid_accuracies)]])
            row_df = pd.DataFrame(index=[job], data=row.reshape(1, -1), columns=columns)
            output_df = pd.concat([output_df, row_df])

        total_df = pd.DataFrame(
            np.array(output_df.sum()).reshape(1, -1), columns=columns, index=["total"]
        )
        output_df = pd.concat([output_df, total_df])
        output_df.sort_index(inplace=True)

        output_df.to_csv(
            path.join(launch_dir, "analysis_" + selection + ".tsv"), sep="\t"
        )
