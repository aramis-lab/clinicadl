import pytest
from torch import Size, Tensor
from torch.nn import MSELoss


def test_get_metric():
    from monai.metrics import SSIMMetric

    from clinicadl.monai_metrics import get_metric
    from clinicadl.monai_metrics.config import ImplementedMetrics, create_metric_config

    for metric_name in [e.value for e in ImplementedMetrics if e != "Loss"]:
        if (
            metric_name == ImplementedMetrics.SSIM
            or metric_name == ImplementedMetrics.MS_SSIM
        ):
            params = {"spatial_dims": 3, "kernel_sigma": 13.0}
        elif metric_name == ImplementedMetrics.PSNR:
            params = {"max_val": 3}
        elif metric_name == ImplementedMetrics.SURF_DICE:
            params = {"class_thresholds": [0.1, 0.2]}
        else:
            params = {}
        config = create_metric_config(metric_name)(**params)

        metric, updated_config = get_metric(config)

        if metric_name == "SSIM":
            assert isinstance(metric, SSIMMetric)
            assert metric.spatial_dims == 3
            assert metric.data_range == 1.0
            assert metric.kernel_type == "gaussian"
            assert metric.kernel_sigma == (13.0, 13.0, 13.0)

            assert updated_config.metric == "SSIMMetric"
            assert updated_config.spatial_dims == 3
            assert updated_config.data_range == 1.0
            assert updated_config.kernel_type == "gaussian"
            assert updated_config.kernel_sigma == 13.0
            assert updated_config.k1 == 0.01


@pytest.mark.skip()
def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum()


@pytest.mark.skip()
def loss_fn_bis(y_pred: Tensor) -> Tensor:
    return (y_pred**2).sum()


def test_loss_to_metric():
    from torch import randn

    from clinicadl.monai_metrics import loss_to_metric

    y_pred = randn(10, 5, 5)
    y_true = randn(10, 5, 5)

    with pytest.raises(ValueError):
        loss_to_metric(loss_fn)

    metric = loss_to_metric(MSELoss(reduction="sum"), reduction="mean")
    assert metric.reduction == "mean"
    assert metric(y_pred, y_true).shape == Size((1, 1))

    metric = loss_to_metric(MSELoss(reduction="sum"))
    assert metric.reduction == "sum"
    assert metric(y_pred, y_true).shape == Size((1, 1))

    metric = loss_to_metric(loss_fn, reduction="sum")
    assert metric.reduction == "sum"
    assert metric(y_pred, y_true).shape == Size((1, 1))

    metric = loss_to_metric(loss_fn_bis, reduction="sum")
    assert metric.reduction == "sum"
    assert metric(y_pred).shape == Size((1, 1))
