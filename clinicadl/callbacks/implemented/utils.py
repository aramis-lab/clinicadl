from __future__ import annotations

import pandas as pd

from clinicadl.utils.dictionary.words import EPOCH


def get_metric_value(metrics_df: pd.DataFrame, metric_name: str, epoch: int) -> float:
    """
    Gets a value from a DataFrame of metrics, like :py:attr:`clinicadl.metrics.handler.MetricsHandler.df`.

    The DataFrame must contain a column named ``"epoch"``, and a column
    named after the wanted metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The DataFrame containing the metric values.
    metric_name : str
        The metric to retrieve.
    epoch : int
        The epoch at which the metric must be obtained.
    """

    if metric_name not in metrics_df:
        raise get_metric_key_error(metric_name)

    value = metrics_df.set_index(EPOCH).loc[epoch, metric_name]

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Value for metric '{metric_name}' at epoch {epoch} is not numeric."
        ) from exc


def get_metric_key_error(metric_name: str) -> KeyError:
    """
    A KeyError for when a desired metric is not in the metrics computed.
    """
    return KeyError(f"'{metric_name}' not found in the validation metrics!")
