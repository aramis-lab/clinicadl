from copy import copy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from clinicadl.tsvtools.utils import first_session
from clinicadl.utils.exceptions import ClinicaDLTSVError


def add_demographics(df, demographics_df, diagnosis) -> pd.DataFrame:
    out_df = pd.DataFrame()
    tmp_demo_df = copy(demographics_df)
    tmp_demo_df.reset_index(inplace=True)
    for idx in df.index.values:
        participant = df.loc[idx, "participant_id"]
        session = df.loc[idx, "session_id"]
        row_df = tmp_demo_df[
            (tmp_demo_df.participant_id == participant)
            & (tmp_demo_df.session_id == session)
        ]
        out_df = pd.concat([out_df, row_df])
    out_df.reset_index(inplace=True, drop=True)
    out_df.diagnosis = [diagnosis] * len(out_df)
    return out_df


def remove_unicity(values_list):
    """Count the values of each class and label all the classes with only one label under the same label."""
    unique_classes, counts = np.unique(values_list, return_counts=True)
    one_sub_classes = unique_classes[(counts == 1)]
    for class_element in one_sub_classes:
        values_list[values_list.index(class_element)] = unique_classes.min()

    return values_list


def category_conversion(values_list) -> List[int]:
    values_np = np.array(values_list)
    unique_classes = np.unique(values_np)
    for index, unique_class in enumerate(unique_classes):
        values_np[values_np == unique_class] = index + 1

    return values_np.astype(int).tolist()


def find_label(labels_list, target_label):
    if target_label in labels_list:
        return target_label
    else:
        min_length = np.inf
        found_label = None
        for label in labels_list:
            if target_label.lower() in label.lower() and min_length > len(label):
                min_length = len(label)
                found_label = label
        if found_label is None:
            raise ClinicaDLTSVError(
                f"No label was found in {labels_list} for target label {target_label}."
            )

        return found_label


def find_splits(maps_path: Path) -> List[int]:
    """Find which splits that were trained in the MAPS."""
    splits = [
        int(split.name.split("-")[1])
        for split in list(maps_path.iterdir())
        if split.name.startswith("split-")
    ]
    return splits


def find_stopped_splits(maps_path: Path) -> List[int]:
    """Find which splits for which training was not completed."""
    existing_split_list = find_splits(maps_path)
    stopped_splits = [
        split
        for split in existing_split_list
        if (maps_path / f"split-{split}" / "tmp")
        in list((maps_path / f"split-{split}").iterdir())
    ]
    return stopped_splits


def find_finished_splits(maps_path: Path) -> List[int]:
    """Find which splits for which training was completed."""
    finished_splits = list()
    existing_split_list = find_splits(maps_path)
    stopped_splits = find_stopped_splits(maps_path)
    for split in existing_split_list:
        if split not in stopped_splits:
            performance_dir_list = [
                performance_dir
                for performance_dir in list((maps_path / f"split-{split}").iterdir())
                if "best-" in performance_dir.name
            ]
            if len(performance_dir_list) > 0:
                finished_splits.append(split)
    return finished_splits


def print_description_log(
    maps_path: Path,
    data_group: str,
    split: int,
    selection_metric: str,
):
    """
    Print the description log associated to a prediction or interpretation.

    Args:
        data_group (str): name of the data group used for the task.
        split (int): Index of the split used for training.
        selection_metric (str): Metric used for best weights selection.
    """
    log_dir = maps_path / f"split-{split}" / f"best-{selection_metric}" / data_group
    log_path = log_dir / "description.log"
    with log_path.open(mode="r") as f:
        content = f.read()
