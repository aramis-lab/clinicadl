import json
import shutil
import subprocess
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.callbacks.callbacks import Callback, CallbacksHandler
from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
    MAPSError,
)
from clinicadl.utils.maps_manager.ddp import DDP, cluster, init_ddp
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.maps_manager.maps_manager import MapsManager
from clinicadl.utils.maps_manager.maps_manager_utils import (
    add_default_values,
    read_json,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.predict_manager.predict_config import PredictConfig
from clinicadl.utils.preprocessing import path_decoder, path_encoder
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything

logger = getLogger("clinicadl.maps_manager")
level_list: List[str] = ["warning", "info", "debug"]


class PredictManager:
    def __init__(self, maps_manager: MapsManager, predict_config):
        self.maps_manager = maps_manager
        # self.predict_config = PredictConfig()

    def predict(
        self,
        data_group: str,
        caps_directory: Path = None,
        tsv_path: Path = None,
        split_list: List[int] = None,
        selection_metrics: List[str] = None,
        multi_cohort: bool = False,
        diagnoses: List[str] = (),
        use_labels: bool = True,
        batch_size: int = None,
        n_proc: int = None,
        gpu: bool = None,
        amp: bool = False,
        overwrite: bool = False,
        label: str = None,
        label_code: Optional[Dict[str, int]] = "default",
        save_tensor: bool = False,
        save_nifti: bool = False,
        save_latent_tensor: bool = False,
        skip_leak_check: bool = False,
    ):
        """
        Performs the prediction task on a subset of caps_directory defined in a TSV file.

        Args:
            data_group: name of the data group tested.
            caps_directory: path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default will load the value of an existing data group
            tsv_path: path to a TSV file containing the list of participants and sessions to test.
                Default will load the DataFrame of an existing data group
            split_list: list of splits to test. Default perform prediction on all splits available.
            selection_metrics (list[str]): list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
            multi_cohort: If True considers that tsv_path is the path to a multi-cohort TSV.
            diagnoses: List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            use_labels: If True, the labels must exist in test meta-data and metrics are computed.
            batch_size: If given, sets the value of batch_size, else use the same as in training step.
            n_proc: If given, sets the value of num_workers, else use the same as in training step.
            gpu: If given, a new value for the device of the model will be computed.
            amp: If enabled, uses Automatic Mixed Precision (requires GPU usage).
            overwrite: If True erase the occurrences of data_group.
            label: Target label used for training (if network_task in [`regression`, `classification`]).
            label_code: dictionary linking the target values to a node number.
        """
        if not split_list:
            split_list = self._find_splits()
        logger.debug(f"List of splits {split_list}")

        _, all_transforms = get_transforms(
            normalize=self.maps_manager.normalize,
            data_augmentation=self.maps_manager.data_augmentation,
            size_reduction=self.maps_manager.size_reduction,
            size_reduction_factor=self.maps_manager.size_reduction_factor,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.maps_manager.diagnoses,
                multi_cohort=multi_cohort,
            )
        criterion = self.maps_manager.task_manager.get_criterion(self.maps_manager.loss)
        self.maps_manager._check_data_group(
            data_group,
            caps_directory,
            group_df,
            multi_cohort,
            overwrite,
            label=label,
            split_list=split_list,
            skip_leak_check=skip_leak_check,
        )
        for split in split_list:
            logger.info(f"Prediction of split {split}")
            group_df, group_parameters = self.maps_manager.get_group_info(
                data_group, split
            )
            # Find label code if not given
            if (
                label is not None
                and label != self.maps_manager.label
                and label_code == "default"
            ):
                self.maps_manager.task_manager.generate_label_code(group_df, label)

            # Erase previous TSV files on master process
            if not selection_metrics:
                split_selection_metrics = self.maps_manager._find_selection_metrics(
                    split
                )
            else:
                split_selection_metrics = selection_metrics
            for selection in split_selection_metrics:
                tsv_dir = (
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{selection}"
                    / data_group
                )

                tsv_pattern = f"{data_group}*.tsv"

                for tsv_file in tsv_dir.glob(tsv_pattern):
                    tsv_file.unlink()

            if self.maps_manager.multi_network:
                self._predict_multi(
                    group_parameters,
                    group_df,
                    all_transforms,
                    use_labels,
                    label,
                    label_code,
                    batch_size,
                    n_proc,
                    criterion,
                    data_group,
                    split,
                    split_selection_metrics,
                    gpu,
                    amp,
                    save_tensor,
                    save_latent_tensor,
                    save_nifti,
                    selection_metrics,
                )

            else:
                self._predict_single(
                    group_parameters,
                    group_df,
                    all_transforms,
                    use_labels,
                    label,
                    label_code,
                    batch_size,
                    n_proc,
                    criterion,
                    data_group,
                    split,
                    split_selection_metrics,
                    gpu,
                    amp,
                    save_tensor,
                    save_latent_tensor,
                    save_nifti,
                    selection_metrics,
                )

            if cluster.master:
                self.maps_manager._ensemble_prediction(
                    data_group, split, selection_metrics, use_labels, skip_leak_check
                )

    def _predict_multi(
        self,
        group_parameters,
        group_df,
        all_transforms,
        use_labels,
        label,
        label_code,
        batch_size,
        n_proc,
        criterion,
        data_group,
        split,
        split_selection_metrics,
        gpu,
        amp,
        save_tensor,
        save_latent_tensor,
        save_nifti,
        selection_metrics,
    ):
        for network in range(self.maps_manager.num_networks):
            data_test = return_dataset(
                group_parameters["caps_directory"],
                group_df,
                self.maps_manager.preprocessing_dict,
                all_transformations=all_transforms,
                multi_cohort=group_parameters["multi_cohort"],
                label_presence=use_labels,
                label=self.maps_manager.label if label is None else label,
                label_code=(
                    self.maps_manager.label_code
                    if label_code == "default"
                    else label_code
                ),
                cnn_index=network,
            )
            test_loader = DataLoader(
                data_test,
                batch_size=(
                    batch_size
                    if batch_size is not None
                    else self.maps_manager.batch_size
                ),
                shuffle=False,
                sampler=DistributedSampler(
                    data_test,
                    num_replicas=cluster.world_size,
                    rank=cluster.rank,
                    shuffle=False,
                ),
                num_workers=n_proc if n_proc is not None else self.maps_manager.n_proc,
            )
            self.maps_manager._test_loader(
                test_loader,
                criterion,
                data_group,
                split,
                split_selection_metrics,
                use_labels=use_labels,
                gpu=gpu,
                amp=amp,
                network=network,
            )
            if save_tensor:
                logger.debug("Saving tensors")
                self.maps_manager._compute_output_tensors(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )
            if save_nifti:
                self.maps_manager._compute_output_nifti(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )
            if save_latent_tensor:
                self.maps_manager._compute_latent_tensors(
                    data_test,
                    data_group,
                    split,
                    selection_metrics,
                    gpu=gpu,
                    network=network,
                )

    def _predict_single(
        self,
        group_parameters,
        group_df,
        all_transforms,
        use_labels,
        label,
        label_code,
        batch_size,
        n_proc,
        criterion,
        data_group,
        split,
        split_selection_metrics,
        gpu,
        amp,
        save_tensor,
        save_latent_tensor,
        save_nifti,
        selection_metrics,
    ):
        data_test = return_dataset(
            group_parameters["caps_directory"],
            group_df,
            self.maps_manager.preprocessing_dict,
            all_transformations=all_transforms,
            multi_cohort=group_parameters["multi_cohort"],
            label_presence=use_labels,
            label=self.maps_manager.label if label is None else label,
            label_code=(
                self.maps_manager.label_code if label_code == "default" else label_code
            ),
        )

        test_loader = DataLoader(
            data_test,
            batch_size=(
                batch_size if batch_size is not None else self.maps_manager.batch_size
            ),
            shuffle=False,
            sampler=DistributedSampler(
                data_test,
                num_replicas=cluster.world_size,
                rank=cluster.rank,
                shuffle=False,
            ),
            num_workers=n_proc if n_proc is not None else self.maps_manager.n_proc,
        )
        self.maps_manager._test_loader(
            test_loader,
            criterion,
            data_group,
            split,
            split_selection_metrics,
            use_labels=use_labels,
            gpu=gpu,
            amp=amp,
        )
        if save_tensor:
            logger.debug("Saving tensors")
            self.maps_manager._compute_output_tensors(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )
        if save_nifti:
            self.maps_manager._compute_output_nifti(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )
        if save_latent_tensor:
            self.maps_manager._compute_latent_tensors(
                data_test,
                data_group,
                split,
                selection_metrics,
                gpu=gpu,
            )
