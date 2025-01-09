import json
from logging import getLogger
from pathlib import Path

import pandas as pd

from clinicadl.utils.exceptions import ClinicaDLArgumentError

logger = getLogger("clinicadl.tsvtools.adapt")


def concat_files(file_path: Path, df_baseline, df_all):
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
    if file_path.name.endswith("baseline.tsv"):
        df_baseline = pd.concat([df_baseline, label_df])
    elif file_path.name.endswith(".tsv"):
        df_all = pd.concat([df_all, label_df])
    return df_all, df_baseline


def adapt(old_tsv_dir: Path, new_tsv_dir: Path, subset_name="labels", labels_list="AD"):
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

    if not old_tsv_dir.is_dir():
        raise ClinicaDLArgumentError(
            f"the directory: {old_tsv_dir} doesn't exists"
            "Please give an existing path"
        )
    if not new_tsv_dir.is_dir():
        new_tsv_dir.mkdir(parents=True, exist_ok=True)

    files_list = list(old_tsv_dir.iterdir())

    df_baseline = pd.DataFrame()
    df_all = pd.DataFrame()

    for file in files_list:
        path = old_tsv_dir / file
        if path.is_file() and path.name.endswith(".tsv"):
            df_all, df_baseline = concat_files(path, df_baseline, df_all)

    if not df_all.empty:
        df_all.to_csv(str(new_tsv_dir / (subset_name + ".tsv")), sep="\t", index=False)

    if not df_baseline.empty:
        df_baseline.to_csv(
            str(new_tsv_dir / (subset_name + "_baseline.tsv")), sep="\t", index=False
        )

    if (old_tsv_dir / "split.json") in files_list:
        new_split_dir = new_tsv_dir / "split"
        with (old_tsv_dir / "split.json").open(mode="r") as f:
            parameters_split = json.load(f)
        subset_name = parameters_split["subset_name"]
        adapt(
            old_tsv_dir / subset_name,
            new_split_dir,
            subset_name,
            labels_list,
        )
        adapt(old_tsv_dir / "train", new_split_dir, "train", labels_list)

    if (old_tsv_dir / "kfold.json") in files_list:
        with (old_tsv_dir / "kfold.json").open(mode="r") as f:
            parameters_kfold = json.load(f)

        subset_name = parameters_kfold["subset_name"]
        n_splits = parameters_kfold["n_splits"]

        new_fold_dir = new_tsv_dir / (str(n_splits) + "_fold")
        new_fold_dir.mkdir(parents=True)

        for i in range(n_splits):
            new_kfold_dir = (
                new_tsv_dir / (str(n_splits) + "_fold") / ("split-" + str(i))
            )

            adapt(
                old_tsv_dir / ("train_splits-" + str(n_splits)) / ("split-" + str(i)),
                new_kfold_dir,
                "train",
                labels_list,
            )
            adapt(
                old_tsv_dir
                / (subset_name + "_splits-" + str(n_splits))
                / ("split-" + str(i)),
                new_kfold_dir,
                subset_name,
                labels_list,
            )

    logger.info(f"New directory was created at {new_tsv_dir}")
