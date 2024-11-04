import pytest
from monai.metrics import SSIMMetric
from torch import Size, Tensor
from torch.nn import MSELoss

from clinicadl.metrics import ImplementedMetric, get_metric, get_metric_from_config
from clinicadl.metrics.config import ImplementedMetric
from clinicadl.metrics.config.reconstruction import SSIMMetricConfig


@pytest.mark.parametrize(
    "metric_name,params",
    [
        (
            ImplementedMetric.SSIM.value,
            {"spatial_dims": 3},
        ),
        (
            ImplementedMetric.MS_SSIM.value,
            {"spatial_dims": 3},
        ),
        (
            ImplementedMetric.PSNR.value,
            {"max_val": 3},
        ),
        (
            ImplementedMetric.SURF_DICE.value,
            {"class_thresholds": [0.1, 0.2]},
        ),
    ]
    + [
        (metric.value, {})
        for metric in ImplementedMetric
        if metric
        not in {
            ImplementedMetric.SSIM,
            ImplementedMetric.MS_SSIM,
            ImplementedMetric.PSNR,
            ImplementedMetric.SURF_DICE,
            ImplementedMetric.LOSS,
        }
    ],
)
def test_get_metric(metric_name, params):
    _ = get_metric(name=metric_name, **params)


def test_parameters():
    metric, config = get_metric(
        name="SSIMMetric",
        spatial_dims=3,
        return_config=True,
        data_range=1.0,
        kernel_type="gaussian",
        kernel_sigma=13.0,
    )
    assert isinstance(metric, SSIMMetric)
    assert metric.spatial_dims == 3
    assert metric.data_range == 1.0
    assert metric.kernel_type == "gaussian"
    assert metric.kernel_sigma == (13.0, 13.0, 13.0)

    assert config.name == "SSIMMetric"
    assert config.spatial_dims == 3
    assert config.data_range == 1.0
    assert config.kernel_type == "gaussian"
    assert config.kernel_sigma == 13.0
    assert config.k1 == 0.01


def test_without_return():
    net = get_metric("SSIMMetric", spatial_dims=3)
    assert isinstance(net, SSIMMetric)


def test_get_loss_function_from_config():
    config = SSIMMetricConfig(spatial_dims=3)
    metric, updated_config = get_metric_from_config(config)
    assert isinstance(metric, SSIMMetric)
    assert updated_config.kernel_sigma == 1.5
    assert config.kernel_sigma == "DefaultFromLibrary"


@pytest.mark.skip()
def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum()


@pytest.mark.skip()
def loss_fn_bis(y_pred: Tensor) -> Tensor:
    return (y_pred**2).sum()


def test_loss_to_metric():
    from torch import randn

    from clinicadl.metrics import loss_to_metric

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
