import json
import os
import shutil
from os.path import join
from pathlib import Path

from train.tasks.task_utils import task_launcher

params = [
    "image_ae",
    "patch_multi_ae",
    "roi_ae",
    "slice_ae",
]

task_specific_options = ["selection_metrics", "loss"]

base_dir = Path("/Users/camille.brianceau/aramis/data_ci")
input_dir = base_dir / "train" / "in"
ref_dir = base_dir / "train" / "ref"
tmp_out_dir = base_dir / "train" / "out8"


_caps_dir = str(input_dir / "caps_image")
_json_path = "t1-linear_crop-True_mode-image.json"
_labels_path = str(input_dir / "labels_list" / "2_fold")
_maps_dir = str(tmp_out_dir)
_config_file = "/Users/camille.brianceau/aramis/clinicadl/clinicadl/config-7.toml"

# import toml

# config_dict = toml.load(_config_file)

## TOML
_architecture = "AE_Conv4_FC3"
_multi_network = False
_multi_cohort = False
_dropout = 0.0
_selection_metrics = ["loss"]
_loss = "L1Loss"
_gpu = False
_n_proc = 2
_batch_size = 8
_evaluation_steps = 0
_seed = 0
_deterministic = False
_transfer_path = ""
_transfer_selection_metric = ""
_use_extracted_features = False
_diagnoses = ["AD", "CN"]
_baseline = False
_normalize = True
_data_augmentation = []
_sampler = "random"
_size_reduction = False
_n_splits = 2
_split = [1]
_optimizer = "Adam"
_epochs = 5
_learning_rate = 0.0001
_weight_decay = 0.0001
_patience = 0
_tolerance = 0.0
_accumulation_steps = 1
_profiler = False
_compensation = False

task_launcher(
    "reconstruction",
    task_specific_options,
    caps_directory=_caps_dir,
    preprocessing_json=_json_path,
    tsv_directory=_labels_path,
    output_maps_directory=_maps_dir,
    config_file=_config_file,
    architecture=_architecture,
    multi_network=_multi_network,
    multi_cohort=_multi_cohort,
    dropout=_dropout,
    selection_metrics=_selection_metrics,
    loss=_loss,
    gpu=_gpu,
    n_proc=_n_proc,
    batch_size=_batch_size,
    evaluation_steps=_evaluation_steps,
    seed=_seed,
    deterministic=_deterministic,
    transfer_path=_transfer_path,
    transfer_selection_metric=_transfer_selection_metric,
    use_extracted_features=_use_extracted_features,
    diagnoses=_diagnoses,
    baseline=_baseline,
    normalize=_normalize,
    data_augmentation=_data_augmentation,
    sampler=_sampler,
    size_reduction=_size_reduction,
    n_splits=_n_splits,
    split=_split,
    optimizer=_optimizer,
    epochs=_epochs,
    learning_rate=_learning_rate,
    weight_decay=_weight_decay,
    patience=_patience,
    tolerance=_tolerance,
    accumulation_steps=_accumulation_steps,
    profiler=_profiler,
    compensation=_compensation,
)
