from pathlib import Path
from typing import Optional

import pandas as pd

from clinicadl.metrics.utils import check_selection_metric
from clinicadl.splitter.split_utils import print_description_log
from clinicadl.utils.exceptions import MAPSError


def get_prediction(
    maps_path: Path,
    data_group: str,
    split: int = 0,
    selection_metric: Optional[str] = None,
    mode: str = "image",
    verbose: bool = False,
):
    """
    Get the individual predictions for each participant corresponding to one group
    of participants identified by its data group.

    Args:
        data_group (str): name of the data group used for the prediction task.
        split (int): Index of the split used for training.
        selection_metric (str): Metric used for best weights selection.
        mode (str): level of the prediction.
        verbose (bool): if True will print associated prediction.log.
    Returns:
        (DataFrame): Results indexed by columns 'participant_id' and 'session_id' which
        identifies the image in the BIDS / CAPS.
    """
    selection_metric = check_selection_metric(maps_path, split, selection_metric)
    if verbose:
        print_description_log(maps_path, data_group, split, selection_metric)
    prediction_dir = (
        maps_path / f"split-{split}" / f"best-{selection_metric}" / data_group
    )
    print(prediction_dir)
    if not prediction_dir.is_dir():
        raise MAPSError(
            f"No prediction corresponding to data group {data_group} was found."
        )
    df = pd.read_csv(
        prediction_dir / f"{data_group}_{mode}_level_prediction.tsv",
        sep="\t",
    )
    df.set_index(["participant_id", "session_id"], inplace=True, drop=True)
    return df
