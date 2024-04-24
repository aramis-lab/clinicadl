import pytest
from pydantic import ValidationError


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "flair-linear",
            "n_subjects": 3,
            "n_proc": 1,
            "pathology_cls": "lvppa",
            "anomaly_degree": 6,
            "sigma": 5,
            "use_uncropped_image": False,
        },
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "t1-linear",
            "n_subjects": 3,
            "n_proc": 1,
            "pathology_cls": "alzheimer",
            "anomaly_degree": 6,
            "sigma": 5,
            "use_uncropped_image": False,
        },
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "t1-linear",
            "n_subjects": 3,
            "n_proc": 1,
            "pathology_cls": "lvppa",
            "anomaly_degree": 6,
            "sigma": 40.2,
            "use_uncropped_image": True,
        },
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "t1-linear",
            "n_subjects": 3,
            "n_proc": 0,
            "pathology_cls": "lvppa",
            "anomaly_degree": 6,
            "sigma": 5,
            "use_uncropped_image": False,
        },
    ],
)
def test_fails_validations(parameters):
    from clinicadl.generate.generate_config import GenerateHypometabolicConfig

    with pytest.raises(ValidationError):
        GenerateHypometabolicConfig(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "t1-linear",
            "n_subjects": 3,
            "n_proc": 2,
            "pathology_cls": "lvppa",
            "anomaly_degree": 30.5,
            "sigma": 35,
            "use_uncropped_image": False,
        },
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "pet-linear",
            "n_subjects": 3,
            "n_proc": 1,
            "pathology_cls": "ad",
            "anomaly_degree": 6.6,
            "sigma": 20,
            "use_uncropped_image": True,
        },
        {
            "caps_directory": "",
            "generated_caps_directory": "",
            "participants_list": "",
            "preprocessing_cls": "t1-linear",
            "n_subjects": 3,
            "n_proc": 1,
            "pathology_cls": "pca",
            "anomaly_degree": 6,
            "sigma": 5,
            "use_uncropped_image": True,
        },
    ],
)
def test_passes_validations(parameters):
    from clinicadl.generate.generate_config import GenerateHypometabolicConfig

    GenerateHypometabolicConfig(**parameters)
