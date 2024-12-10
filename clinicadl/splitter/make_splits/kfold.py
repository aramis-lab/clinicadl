import json
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple, Union

from pydantic import NonNegativeInt, PositiveInt
from sklearn.model_selection import StratifiedKFold

from clinicadl.dataset.utils import tsv_to_df
from clinicadl.splitter.make_splits.utils import (
    _write_to_csv,
    preprocess_stratification,
)
from clinicadl.splitter.splitter.kfold import KFoldConfig
from clinicadl.tsvtools.tsvtools_utils import extract_baseline, retrieve_longitudinal


def make_kfold(
    tsv_path: Path,
    output_dir: Optional[Path] = None,
    subset_name: str = "validation",
    valid_longitudinal: bool = False,
    n_splits: PositiveInt = 5,
    stratification: Optional[List[str]] = None,
    ignore_demographics: bool = False,
) -> Path:
    """
    Perform K-Fold splitting with optional stratification.

    Parameters
    ----------
    n_splits : PositiveInt, default=5
        Number of splits.
    stratification : Optional[List[str]]
        Columns to use for stratification.
    ignore_demographics : bool, default=False
        If True, demographic balancing is ignored.

    Returns
    -------
    None
        Populates the `subjects_sessions_split` attribute with the generated splits.
    """

    if not output_dir:
        output_dir = tsv_path.parent

    config = KFoldConfig(
        split_dir=output_dir,
        subset_name=subset_name,
        valid_longitudinal=valid_longitudinal,
        n_splits=n_splits,
        stratification=stratification,
        ignore_demographics=ignore_demographics,
    )

    config._check_split_dir()
    config._write_json()

    df = tsv_to_df(tsv_path)
    baseline_df = extract_baseline(df)

    stratify_labels = preprocess_stratification(
        df=baseline_df,
        columns=config.stratification,
        ignore_demographics=config.ignore_demographics,
    )
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=2)

    for i, (train_idx, test_idx) in enumerate(skf.split(baseline_df, stratify_labels)):
        train = baseline_df.iloc[train_idx]
        test = baseline_df.iloc[test_idx]

        split_dir = config.split_dir / f"split-{i}"
        split_dir.mkdir(parents=True, exist_ok=True)

        _write_to_csv(train, split_dir / "train_baseline.tsv")
        _write_to_csv(test, split_dir / f"{config.subset_name}_baseline.tsv")

        long_train_df = retrieve_longitudinal(train, df)
        _write_to_csv(long_train_df, split_dir / "train.tsv")

        if config.valid_longitudinal:
            long_val_df = retrieve_longitudinal(test, df)
            _write_to_csv(long_val_df, split_dir / f"{subset_name}.tsv")

    return config.split_dir
