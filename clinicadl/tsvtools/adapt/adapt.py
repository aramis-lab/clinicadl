import json
import os
from logging import getLogger

import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl")


def concat_files(file_path, df_baseline, df_all):
    """
    Read a file and concatenates it to the right dataframe (baseline or not).

    Parameters
    ----------
    file_path: str (path)
        Path to the TSV file (with "participant_id" and "session_id" in the columns).
    df_baseline: DataFrame
        Dataframe with only baseline sessions.
    df_all: DataFrame
        Dataframe with all sessions.
    """

    label_df = pd.read_csv(file_path, sep="\t")
    if file_path.endswith("baseline.tsv"):
        df_baseline = pd.concat([df_baseline, label_df])
    elif file_path.endswith(".tsv"):
        df_all = pd.concat([df_all, label_df])
    return df_all, df_baseline


def adapt(old_tsv_dir: str, new_tsv_dir: str, subset_name="labels", labels_list="AD"):
    """
    Produces a new split/kfold directories that fit with clinicaDL 1.2.0.

    Parameters
    ----------
    old_tsv_dir: str (path)
        Path to the old directory.
    new_tsv_dir: str (path)
        Path to the fnew directory.
    subset_name: str
        Name of the output file of `clinicadl get-labels`.
    labels_list: list of str
        list of labels (in the old way, each labels had its own TSV file).

    """

    if not os.path.exists(old_tsv_dir):
        raise ClinicaDLArgumentError(
            f"the directory: {old_tsv_dir} doesn't exists"
            "Please give an existing path"
        )
    if not os.path.exists(new_tsv_dir):
        os.makedirs(new_tsv_dir, exist_ok=True)

    files_list = os.listdir(old_tsv_dir)

    df_baseline = pd.DataFrame()
    df_all = pd.DataFrame()

    for file in files_list:
        path = os.path.join(old_tsv_dir, file)
        if os.path.isfile(path) and path.endswith(".tsv"):
            df_all, df_baseline = concat_files(path, df_baseline, df_all)

    if not df_all.empty:
        df_all.to_csv(os.path.join(new_tsv_dir, subset_name + ".tsv"), sep="\t")
    if not df_baseline.empty:
        df_baseline.to_csv(
            os.path.join(new_tsv_dir, subset_name + "_baseline.tsv"), sep="\t"
        )

    if "split.json" in files_list:
        new_split_dir = os.path.join(new_tsv_dir, "split")
        with open(os.path.join(old_tsv_dir, "split.json"), "r") as f:
            parameters_split = json.load(f)
        subset_name = parameters_split["subset_name"]
        adapt(
            os.path.join(old_tsv_dir, subset_name),
            new_split_dir,
            subset_name,
            labels_list,
        )
        adapt(os.path.join(old_tsv_dir, "train"), new_split_dir, "train", labels_list)

    if "kfold.json" in files_list:
        with open(os.path.join(old_tsv_dir, "kfold.json"), "r") as f:
            parameters_kfold = json.load(f)

        subset_name = parameters_kfold["subset_name"]
        n_splits = parameters_kfold["n_splits"]

        new_fold_dir = os.path.join(new_tsv_dir, str(n_splits) + "_fold")
        os.makedirs(new_fold_dir)

        for i in range(n_splits):
            new_kfold_dir = os.path.join(
                new_tsv_dir, str(n_splits) + "_fold", "split-" + str(i)
            )
            adapt(
                os.path.join(
                    old_tsv_dir, "train_splits-" + str(n_splits), "split-" + str(i)
                ),
                new_kfold_dir,
                "train",
                labels_list,
            )
            adapt(
                os.path.join(
                    old_tsv_dir,
                    subset_name + "_splits-" + str(n_splits),
                    "split-" + str(i),
                ),
                new_kfold_dir,
                subset_name,
                labels_list,
            )

    logger.info(f"New directory was created at {new_tsv_dir}")
