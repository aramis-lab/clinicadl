import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "abc",
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "MSELoss",
            "selection_metrics": "loss",
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.train.tasks.regression_config import RegressionConfig

    with pytest.raises(ValidationError):
        RegressionConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "MSELoss",
            "selection_metrics": ("loss",),
        },
        {
            "caps_directory": "",
            "preprocessing_json": "",
            "tsv_directory": "",
            "output_maps_directory": "",
            "loss": "HuberLoss",
            "selection_metrics": ["loss"],
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.train.tasks.regression_config import RegressionConfig

    RegressionConfig(**parameters)
