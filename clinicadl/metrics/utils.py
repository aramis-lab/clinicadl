from clinicadl.metrics.config.factory import ConfusionMatrixMetricConfig


def metric_config_equals(metrics1: list, metrics2: list):
    """
    TO COMPLETE
    If 2 metrics with same name have different parameters return False ???
    """
    for metric1 in metrics1:
        if type(metric1) not in [type(metric2) for metric2 in metrics2]:
            if not isinstance(metric1, ConfusionMatrixMetricConfig):
                return False

    for metric2 in metrics2:
        if type(metric2) not in [type(metric1) for metric1 in metrics1]:
            if not isinstance(metric2, ConfusionMatrixMetricConfig):
                return False

    return True
