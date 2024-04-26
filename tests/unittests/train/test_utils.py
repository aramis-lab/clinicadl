from pathlib import Path

import pytest

from clinicadl.train.tasks import Task

expected_classification = {
    "architecture": "default",
    "multi_network": False,
    "ssda_network": False,
    "dropout": 0.0,
    "latent_space_size": 128,
    "feature_size": 1024,
    "n_conv": 4,
    "io_layer_channels": 8,
    "recons_weight": 1,
    "kl_weight": 1,
    "normalization": "batch",
    "selection_metrics": ["loss"],
    "label": "diagnosis",
    "label_code": {},
    "selection_threshold": 0.0,
    "loss": "CrossEntropyLoss",
    "gpu": True,
    "n_proc": 2,
    "batch_size": 8,
    "evaluation_steps": 0,
    "fully_sharded_data_parallel": False,
    "amp": False,
    "seed": 0,
    "deterministic": False,
    "compensation": "memory",
    "track_exp": "",
    "transfer_path": False,
    "transfer_selection_metric": "loss",
    "nb_unfrozen_layer": 0,
    "use_extracted_features": False,
    "multi_cohort": False,
    "diagnoses": ["AD", "CN"],
    "baseline": False,
    "valid_longitudinal": False,
    "normalize": True,
    "data_augmentation": [],
    "sampler": "random",
    "size_reduction": False,
    "size_reduction_factor": 2,
    "caps_target": "",
    "tsv_target_lab": "",
    "tsv_target_unlab": "",
    "preprocessing_dict_target": "",
    "n_splits": 0,
    "split": [],
    "optimizer": "Adam",
    "epochs": 20,
    "learning_rate": 1e-4,
    "adaptive_learning_rate": False,
    "weight_decay": 1e-4,
    "patience": 0,
    "tolerance": 0.0,
    "accumulation_steps": 1,
    "profiler": False,
    "save_all_models": False,
    "emissions_calculator": False,
}
expected_regression = {
    "architecture": "default",
    "multi_network": False,
    "ssda_network": False,
    "dropout": 0.0,
    "latent_space_size": 128,
    "feature_size": 1024,
    "n_conv": 4,
    "io_layer_channels": 8,
    "recons_weight": 1,
    "kl_weight": 1,
    "normalization": "batch",
    "selection_metrics": ["loss"],
    "label": "age",
    "loss": "MSELoss",
    "gpu": True,
    "n_proc": 2,
    "batch_size": 8,
    "evaluation_steps": 0,
    "fully_sharded_data_parallel": False,
    "amp": False,
    "seed": 0,
    "deterministic": False,
    "compensation": "memory",
    "track_exp": "",
    "transfer_path": False,
    "transfer_selection_metric": "loss",
    "nb_unfrozen_layer": 0,
    "use_extracted_features": False,
    "multi_cohort": False,
    "diagnoses": ["AD", "CN"],
    "baseline": False,
    "valid_longitudinal": False,
    "normalize": True,
    "data_augmentation": [],
    "sampler": "random",
    "size_reduction": False,
    "size_reduction_factor": 2,
    "caps_target": "",
    "tsv_target_lab": "",
    "tsv_target_unlab": "",
    "preprocessing_dict_target": "",
    "n_splits": 0,
    "split": [],
    "optimizer": "Adam",
    "epochs": 20,
    "learning_rate": 1e-4,
    "adaptive_learning_rate": False,
    "weight_decay": 1e-4,
    "patience": 0,
    "tolerance": 0.0,
    "accumulation_steps": 1,
    "profiler": False,
    "save_all_models": False,
    "emissions_calculator": False,
}
expected_reconstruction = {
    "architecture": "default",
    "multi_network": False,
    "ssda_network": False,
    "dropout": 0.0,
    "latent_space_size": 128,
    "feature_size": 1024,
    "n_conv": 4,
    "io_layer_channels": 8,
    "recons_weight": 1,
    "kl_weight": 1,
    "normalization": "batch",
    "selection_metrics": ["loss"],
    "loss": "MSELoss",
    "gpu": True,
    "n_proc": 2,
    "batch_size": 8,
    "evaluation_steps": 0,
    "fully_sharded_data_parallel": False,
    "amp": False,
    "seed": 0,
    "deterministic": False,
    "compensation": "memory",
    "track_exp": "",
    "transfer_path": False,
    "transfer_selection_metric": "loss",
    "nb_unfrozen_layer": 0,
    "use_extracted_features": False,
    "multi_cohort": False,
    "diagnoses": ["AD", "CN"],
    "baseline": False,
    "valid_longitudinal": False,
    "normalize": True,
    "data_augmentation": [],
    "sampler": "random",
    "size_reduction": False,
    "size_reduction_factor": 2,
    "caps_target": "",
    "tsv_target_lab": "",
    "tsv_target_unlab": "",
    "preprocessing_dict_target": "",
    "n_splits": 0,
    "split": [],
    "optimizer": "Adam",
    "epochs": 20,
    "learning_rate": 1e-4,
    "adaptive_learning_rate": False,
    "weight_decay": 1e-4,
    "patience": 0,
    "tolerance": 0.0,
    "accumulation_steps": 1,
    "profiler": False,
    "save_all_models": False,
    "emissions_calculator": False,
}
clinicadl_root_dir = Path(__file__).parents[3] / "clinicadl"
config_toml = clinicadl_root_dir / "resources" / "config" / "train_config.toml"


@pytest.mark.parametrize(
    "config_file,task,expected_output",
    [
        (config_toml, Task.CLASSIFICATION, expected_classification),
        (config_toml, Task.REGRESSION, expected_regression),
        (config_toml, Task.RECONSTRUCTION, expected_reconstruction),
    ],
)
def test_extract_config_from_toml_file(config_file, task, expected_output):
    from clinicadl.train.utils import extract_config_from_toml_file

    assert extract_config_from_toml_file(config_file, task) == expected_output


def test_extract_config_from_toml_file_exceptions():
    from clinicadl.train.utils import extract_config_from_toml_file
    from clinicadl.utils.exceptions import ClinicaDLConfigurationError

    with pytest.raises(ClinicaDLConfigurationError):
        extract_config_from_toml_file(
            Path(str(config_toml).replace(".toml", ".json")),
            Task.CLASSIFICATION,
        )


def test_preprocessing_json_reader():  # TODO : add more test on this function
    from copy import deepcopy

    from clinicadl.train.tasks import BaseTaskConfig
    from clinicadl.train.utils import preprocessing_json_reader

    preprocessing_path = "preprocessing.json"
    config = BaseTaskConfig(
        caps_directory=Path(__file__).parents[3]
        / "tests"
        / "unittests"
        / "train"
        / "ressources"
        / "caps_example",
        preprocessing_json=preprocessing_path,
        tsv_directory="",
        output_maps_directory="",
    )
    expected_config = deepcopy(config)
    expected_config._preprocessing_dict = {
        "preprocessing": "t1-linear",
        "mode": "image",
        "use_uncropped_image": False,
        "prepare_dl": False,
        "extract_json": "t1-linear_mode-image.json",
        "file_type": {
            "pattern": "*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "description": "T1W Image registered using t1-linear and cropped (matrix size 169\u00d7208\u00d7179, 1 mm isotropic voxels)",
            "needed_pipeline": "t1-linear",
        },
    }
    expected_config._mode = "image"

    output_config = preprocessing_json_reader(config)
    assert output_config == expected_config


def test_merge_cli_and_config_file_options():
    import click
    from click.testing import CliRunner

    from clinicadl.train.utils import merge_cli_and_config_file_options

    @click.command()
    @click.option("--config_file")
    @click.option("--compensation", default="default")
    @click.option("--sampler", default="default")
    @click.option("--optimizer", default="default")
    def cli_test(**kwargs):
        return merge_cli_and_config_file_options(Task.CLASSIFICATION, **kwargs)

    config_file = (
        Path(__file__).parents[3]
        / "tests"
        / "unittests"
        / "train"
        / "ressources"
        / "config_example.toml"
    )
    excpected_output = {
        "compensation": "given by user",
        "sampler": "found in config file",
    }

    runner = CliRunner()
    result = runner.invoke(
        cli_test,
        ["--config_file", config_file, "--compensation", "given by user"],
        standalone_mode=False,
    )
    assert result.return_value == excpected_output
