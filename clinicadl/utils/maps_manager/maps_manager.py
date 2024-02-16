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
from clinicadl.utils.logger import setup_logging
from clinicadl.utils.maps_manager.ddp import DDP, cluster, init_ddp
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.maps_manager.maps_manager_utils import (
    add_default_values,
    change_path_to_str,
    change_str_to_path,
    read_json,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything

logger = getLogger("clinicadl.maps_manager")
level_list: List[str] = ["warning", "info", "debug"]


# TODO save weights on CPU for better compatibility


class MapsManager:
    def __init__(
        self,
        maps_path: Path,
        parameters: Dict[str, Any] = None,
        verbose: str = "info",
    ):
        """

        Parameters
        ----------
        maps_path: str (path)
            Path of the MAPS
        parameters: Dict[str, Any]
            Parameters of the training step. If given a new MAPS is created.
        verbose: str
            Logging level ("debug", "info", "warning")
        """
        self.maps_path = maps_path.resolve()

        # Existing MAPS
        if parameters is None:
            if not (maps_path / "maps.json").is_file():
                raise MAPSError(
                    f"MAPS was not found at {maps_path}."
                    f"To initiate a new MAPS please give a train_dict."
                )
            test_parameters = self.get_parameters()
            test_parameters = change_str_to_path(test_parameters)
            self.parameters = add_default_values(test_parameters)
            self.ssda_network = False  # A MODIFIER
            self.save_all_models = self.parameters["save_all_models"]
            self.task_manager = self._init_task_manager(n_classes=self.output_size)
            self.split_name = (
                self._check_split_wording()
            )  # Used only for retro-compatibility

        # Initiate MAPS
        else:
            self._check_args(parameters)
            parameters["tsv_path"] = Path(parameters["tsv_path"])

            self.split_name = "split"  # Used only for retro-compatibility
            if cluster.master:
                if (maps_path.is_dir() and maps_path.is_file()) or (  # Non-folder file
                    maps_path.is_dir() and list(maps_path.iterdir())  # Non empty folder
                ):
                    raise MAPSError(
                        f"You are trying to create a new MAPS at {maps_path} but "
                        f"this already corresponds to a file or a non-empty folder. \n"
                        f"Please remove it or choose another location."
                    )
                (maps_path / "groups").mkdir(parents=True)

                logger.info(f"A new MAPS was created at {maps_path}")

                self.write_parameters(self.maps_path, self.parameters)
                self._write_requirements_version()
                self._write_training_data()
                self._write_train_val_groups()
                self._write_information()

        init_ddp(gpu=self.parameters["gpu"], logger=logger)

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"'MapsManager' object has no attribute '{name}'")

    def train(self, split_list: List[int] = None, overwrite: bool = False):
        """
        Performs the training task for a defined list of splits

        Parameters
        ----------
        split_list: List[int]
            list of splits on which the training task is performed.
            Default trains all splits of the cross-validation.
        overwrite: bool
            If True previously trained splits that are going to be trained are erased.

        Raises
        ------
        Raises MAPSError, if splits specified in input already exist and overwrite is False.
        """
        existing_splits = []

        split_manager = self._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            split_path = self.maps_path / f"{self.split_name}-{split}"
            if split_path.is_dir():
                if overwrite:
                    if cluster.master:
                        shutil.rmtree(split_path)
                else:
                    existing_splits.append(split)

        if len(existing_splits) > 0:
            raise MAPSError(
                f"Splits {existing_splits} already exist. Please "
                f"specify a list of splits not intersecting the previous list, "
                f"or use overwrite to erase previously trained splits."
            )

        if self.multi_network:
            self._train_multi(split_list, resume=False)
        elif self.ssda_network:
            self._train_ssda(split_list, resume=False)
        else:
            self._train_single(split_list, resume=False)

    def resume(self, split_list: List[int] = None):
        """
        Resumes the training task for a defined list of splits.

        Args:
            split_list: list of splits on which the training task is performed.
                Default trains all splits.

        Raises:
            MAPSError: If splits specified in input do not exist.
        """
        missing_splits = []
        split_manager = self._init_split_manager(split_list)

        for split in split_manager.split_iterator():
            if not (self.maps_path / f"{self.split_name}-{split}" / "tmp").is_dir():
                missing_splits.append(split)

        if len(missing_splits) > 0:
            raise MAPSError(
                f"Splits {missing_splits} were not initialized. "
                f"Please try train command on these splits and resume only others."
            )

        if self.multi_network:
            self._train_multi(split_list, resume=True)
        elif self.ssda_network:
            self._train_ssda(split_list, resume=True)
        else:
            self._train_single(split_list, resume=True)

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
            normalize=self.normalize,
            data_augmentation=self.data_augmentation,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.diagnoses,
                multi_cohort=multi_cohort,
            )
        criterion = self.task_manager.get_criterion(self.loss)
        self._check_data_group(
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
            group_df, group_parameters = self.get_group_info(data_group, split)
            # Find label code if not given
            if label is not None and label != self.label and label_code == "default":
                self.task_manager.generate_label_code(group_df, label)

            # Erase previous TSV files on master process
            if not selection_metrics:
                split_selection_metrics = self._find_selection_metrics(split)
            else:
                split_selection_metrics = selection_metrics
            for selection in split_selection_metrics:
                tsv_dir = (
                    self.maps_path
                    / f"{self.split_name}-{split}"
                    / f"best-{selection}"
                    / data_group
                )

                tsv_pattern = f"{data_group}*.tsv"

                for tsv_file in tsv_dir.glob(tsv_pattern):
                    tsv_file.unlink()

            if self.multi_network:
                for network in range(self.num_networks):
                    data_test = return_dataset(
                        group_parameters["caps_directory"],
                        group_df,
                        self.preprocessing_dict,
                        all_transformations=all_transforms,
                        multi_cohort=group_parameters["multi_cohort"],
                        label_presence=use_labels,
                        label=self.label if label is None else label,
                        label_code=(
                            self.label_code if label_code == "default" else label_code
                        ),
                        cnn_index=network,
                    )
                    test_loader = DataLoader(
                        data_test,
                        batch_size=(
                            batch_size if batch_size is not None else self.batch_size
                        ),
                        shuffle=False,
                        sampler=DistributedSampler(
                            data_test,
                            num_replicas=cluster.world_size,
                            rank=cluster.rank,
                            shuffle=False,
                        ),
                        num_workers=n_proc if n_proc is not None else self.n_proc,
                    )
                    self._test_loader(
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
                        self._compute_output_tensors(
                            data_test,
                            data_group,
                            split,
                            selection_metrics,
                            gpu=gpu,
                            network=network,
                        )
                    if save_nifti:
                        self._compute_output_nifti(
                            data_test,
                            data_group,
                            split,
                            selection_metrics,
                            gpu=gpu,
                            network=network,
                        )
                    if save_latent_tensor:
                        self._compute_latent_tensors(
                            data_test,
                            data_group,
                            split,
                            selection_metrics,
                            gpu=gpu,
                            network=network,
                        )
            else:
                data_test = return_dataset(
                    group_parameters["caps_directory"],
                    group_df,
                    self.preprocessing_dict,
                    all_transformations=all_transforms,
                    multi_cohort=group_parameters["multi_cohort"],
                    label_presence=use_labels,
                    label=self.label if label is None else label,
                    label_code=(
                        self.label_code if label_code == "default" else label_code
                    ),
                )

                test_loader = DataLoader(
                    data_test,
                    batch_size=(
                        batch_size if batch_size is not None else self.batch_size
                    ),
                    shuffle=False,
                    sampler=DistributedSampler(
                        data_test,
                        num_replicas=cluster.world_size,
                        rank=cluster.rank,
                        shuffle=False,
                    ),
                    num_workers=n_proc if n_proc is not None else self.n_proc,
                )
                self._test_loader(
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
                    self._compute_output_tensors(
                        data_test,
                        data_group,
                        split,
                        selection_metrics,
                        gpu=gpu,
                    )
                if save_nifti:
                    self._compute_output_nifti(
                        data_test,
                        data_group,
                        split,
                        selection_metrics,
                        gpu=gpu,
                    )
                if save_latent_tensor:
                    self._compute_latent_tensors(
                        data_test,
                        data_group,
                        split,
                        selection_metrics,
                        gpu=gpu,
                    )

            if cluster.master:
                self._ensemble_prediction(
                    data_group, split, selection_metrics, use_labels, skip_leak_check
                )

    def interpret(
        self,
        data_group,
        name,
        method,
        caps_directory: Path = None,
        tsv_path: Path = None,
        split_list=None,
        selection_metrics=None,
        multi_cohort=False,
        diagnoses=(),
        target_node=0,
        save_individual=False,
        batch_size=None,
        n_proc=None,
        gpu=None,
        amp=False,
        overwrite=False,
        overwrite_name=False,
        level=None,
        save_nifti=False,
    ):
        """
        Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.

        Parameters
        ----------
        data_group: str
            Name of the data group interpreted.
        name: str
            Name of the interpretation procedure.
        method: str
            Method used for extraction (ex: gradients, grad-cam...).
        caps_directory: str (Path)
            Path to the CAPS folder. For more information please refer to
            [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
            Default will load the value of an existing data group.
        tsv_path: str (Path)
            Path to a TSV file containing the list of participants and sessions to test.
            Default will load the DataFrame of an existing data group.
        split_list: list of int
            List of splits to interpret. Default perform interpretation on all splits available.
        selection_metrics: list of str
            List of selection metrics to interpret.
            Default performs the interpretation on all selection metrics available.
        multi_cohort: bool
            If True considers that tsv_path is the path to a multi-cohort TSV.
        diagnoses: list of str
            List of diagnoses to load if tsv_path is a split_directory.
            Default uses the same as in training step.
        target_node: int
            Node from which the interpretation is computed.
        save_individual: bool
            If True saves the individual map of each participant / session couple.
        batch_size: int
            If given, sets the value of batch_size, else use the same as in training step.
        n_proc: int
            If given, sets the value of num_workers, else use the same as in training step.
        gpu: bool
            If given, a new value for the device of the model will be computed.
        amp: bool
            If enabled, uses Automatic Mixed Precision (requires GPU usage).
        overwrite: bool
            If True erase the occurrences of data_group.
        overwrite_name: bool
            If True erase the occurrences of name.
        level: int
            Layer number in the convolutional part after which the feature map is chosen.
        save_nifi : bool
            If True, save the interpretation map in nifti format.
        """

        from clinicadl.interpret.gradients import method_dict

        if method not in method_dict.keys():
            raise NotImplementedError(
                f"Interpretation method {method} is not implemented. "
                f"Please choose in {method_dict.keys()}"
            )

        if not split_list:
            split_list = self._find_splits()
        logger.debug(f"List of splits {split_list}")

        if self.multi_network:
            raise NotImplementedError(
                "The interpretation of multi-network framework is not implemented."
            )

        _, all_transforms = get_transforms(
            normalize=self.normalize,
            data_augmentation=self.data_augmentation,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.diagnoses,
                multi_cohort=multi_cohort,
            )
        self._check_data_group(
            data_group, caps_directory, group_df, multi_cohort, overwrite
        )

        for split in split_list:
            logger.info(f"Interpretation of split {split}")
            df_group, parameters_group = self.get_group_info(data_group, split)

            data_test = return_dataset(
                parameters_group["caps_directory"],
                df_group,
                self.preprocessing_dict,
                all_transformations=all_transforms,
                multi_cohort=parameters_group["multi_cohort"],
                label_presence=False,
                label_code=self.label_code,
                label=self.label,
            )

            test_loader = DataLoader(
                data_test,
                batch_size=batch_size if batch_size is not None else self.batch_size,
                shuffle=False,
                num_workers=n_proc if n_proc is not None else self.n_proc,
            )

            if not selection_metrics:
                selection_metrics = self._find_selection_metrics(split)

            for selection_metric in selection_metrics:
                logger.info(f"Interpretation of metric {selection_metric}")
                results_path = (
                    self.maps_path
                    / f"{self.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                    / f"interpret-{name}"
                )

                if (results_path).is_dir():
                    if overwrite_name:
                        shutil.rmtree(results_path)
                    else:
                        raise MAPSError(
                            f"Interpretation name {name} is already written. "
                            f"Please choose another name or set overwrite_name to True."
                        )
                results_path.mkdir(parents=True)

                model, _ = self._init_model(
                    transfer_path=self.maps_path,
                    split=split,
                    transfer_selection=selection_metric,
                    gpu=gpu,
                )

                interpreter = method_dict[method](model)

                cum_maps = [0] * data_test.elem_per_image
                for data in test_loader:
                    images = data["image"].to(model.device)

                    map_pt = interpreter.generate_gradients(
                        images, target_node, level=level, amp=amp
                    )
                    for i in range(len(data["participant_id"])):
                        mode_id = data[f"{self.mode}_id"][i]
                        cum_maps[mode_id] += map_pt[i]
                        if save_individual:
                            single_path = (
                                results_path
                                / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.mode}-{data[f'{self.mode}_id'][i]}_map.pt"
                            )
                            torch.save(map_pt[i], single_path)
                            if save_nifti:
                                import nibabel as nib
                                from numpy import eye

                                single_nifti_path = (
                                    results_path
                                    / f"{data['participant_id'][i]}_{data['session_id'][i]}_{self.mode}-{data[f'{self.mode}_id'][i]}_map.nii.gz"
                                )

                                output_nii = nib.Nifti1Image(map_pt[i].numpy(), eye(4))
                                nib.save(output_nii, single_nifti_path)

                for i, mode_map in enumerate(cum_maps):
                    mode_map /= len(data_test)

                    torch.save(
                        mode_map,
                        results_path / f"mean_{self.mode}-{i}_map.pt",
                    )
                    if save_nifti:
                        import nibabel as nib
                        from numpy import eye

                        output_nii = nib.Nifti1Image(mode_map, eye(4))
                        nib.save(
                            output_nii,
                            results_path / f"mean_{self.mode}-{i}_map.nii.gz",
                        )

    ###################################
    # High-level functions templates  #
    ###################################
    def _train_single(self, split_list=None, resume=False):
        """
        Trains a single CNN for all inputs.

        Args:
            split_list (list[int]): list of splits that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        train_transforms, all_transforms = get_transforms(
            normalize=self.normalize,
            data_augmentation=self.data_augmentation,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )
        split_manager = self._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(self.seed, self.deterministic, self.compensation)

            split_df_dict = split_manager[split]

            logger.debug("Loading training data...")
            data_train = return_dataset(
                self.caps_directory,
                split_df_dict["train"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )
            logger.debug("Loading validation data...")
            data_valid = return_dataset(
                self.caps_directory,
                split_df_dict["validation"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )
            train_sampler = self.task_manager.generate_sampler(
                data_train,
                self.sampler,
                dp_degree=cluster.world_size,
                rank=cluster.rank,
            )
            logger.debug(
                f"Getting train and validation loader with batch size {self.batch_size}"
            )
            train_loader = DataLoader(
                data_train,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.n_proc,
                worker_init_fn=pl_worker_init_function,
            )
            logger.debug(f"Train loader size is {len(train_loader)}")
            valid_sampler = DistributedSampler(
                data_valid,
                num_replicas=cluster.world_size,
                rank=cluster.rank,
                shuffle=False,
            )
            valid_loader = DataLoader(
                data_valid,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_proc,
                sampler=valid_sampler,
            )
            logger.debug(f"Validation loader size is {len(valid_loader)}")
            from clinicadl.utils.callbacks.callbacks import CodeCarbonTracker

            self._train(
                train_loader,
                valid_loader,
                split,
                resume=resume,
                callbacks=[CodeCarbonTracker],
            )

            if cluster.master:
                self._ensemble_prediction(
                    "train",
                    split,
                    self.selection_metrics,
                )
                self._ensemble_prediction(
                    "validation",
                    split,
                    self.selection_metrics,
                )

                self._erase_tmp(split)

    def _train_multi(self, split_list: List[int] = None, resume: bool = False):
        """
        Trains a single CNN per element in the image.

        Args:
            split_list: list of splits that are trained.
            resume: If True the job is resumed from checkpoint.
        """
        train_transforms, all_transforms = get_transforms(
            normalize=self.normalize,
            data_augmentation=self.data_augmentation,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        split_manager = self._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(self.seed, self.deterministic, self.compensation)

            split_df_dict = split_manager[split]

            first_network = 0
            if resume:
                training_logs = [
                    int(network_folder.split("-")[1])
                    for network_folder in list(
                        (
                            self.maps_path
                            / f"{self.split_name}-{split}"
                            / "training_logs"
                        ).iterdir()
                    )
                ]
                first_network = max(training_logs)
                if not (self.maps_path / "tmp").is_dir():
                    first_network += 1
                    resume = False

            for network in range(first_network, self.num_networks):
                logger.info(f"Train network {network}")

                data_train = return_dataset(
                    self.caps_directory,
                    split_df_dict["train"],
                    self.preprocessing_dict,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                    cnn_index=network,
                )
                data_valid = return_dataset(
                    self.caps_directory,
                    split_df_dict["validation"],
                    self.preprocessing_dict,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                    cnn_index=network,
                )

                train_sampler = self.task_manager.generate_sampler(
                    data_train,
                    self.sampler,
                    dp_degree=cluster.world_size,
                    rank=cluster.rank,
                )
                train_loader = DataLoader(
                    data_train,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                    num_workers=self.n_proc,
                    worker_init_fn=pl_worker_init_function,
                )

                valid_sampler = DistributedSampler(
                    data_valid,
                    num_replicas=cluster.world_size,
                    rank=cluster.rank,
                    shuffle=False,
                )
                valid_loader = DataLoader(
                    data_valid,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.n_proc,
                    sampler=valid_sampler,
                )
                from clinicadl.utils.callbacks.callbacks import CodeCarbonTracker

                self._train(
                    train_loader,
                    valid_loader,
                    split,
                    network,
                    resume=resume,
                    callbacks=[CodeCarbonTracker],
                )
                resume = False

            if cluster.master:
                self._ensemble_prediction(
                    "train",
                    split,
                    self.selection_metrics,
                )
                self._ensemble_prediction(
                    "validation",
                    split,
                    self.selection_metrics,
                )

                self._erase_tmp(split)

    def _train_ssda(self, split_list=None, resume=False):
        """
        Trains a single CNN for a source and target domain using semi-supervised domain adaptation.

        Args:
            split_list (list[int]): list of splits that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        from torch.utils.data import DataLoader

        train_transforms, all_transforms = get_transforms(
            normalize=self.normalize,
            data_augmentation=self.data_augmentation,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        split_manager = self._init_split_manager(split_list)

        split_manager_target_lab = self._init_split_manager_ssda(
            self.caps_target, self.tsv_target_lab, split_list
        )

        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(self.seed, self.deterministic, self.compensation)

            split_df_dict = split_manager[split]
            split_df_dict_target_lab = split_manager_target_lab[split]

            logger.debug("Loading source training data...")
            data_train_source = return_dataset(
                self.caps_directory,
                split_df_dict["train"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )

            logger.debug("Loading target labelled training data...")
            data_train_target_labeled = return_dataset(
                Path(self.caps_target),  # TO CHECK
                split_df_dict_target_lab["train"],
                self.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,  # A checker
                label=self.label,
                label_code=self.label_code,
            )
            from torch.utils.data import ConcatDataset, DataLoader

            combined_dataset = ConcatDataset(
                [data_train_source, data_train_target_labeled]
            )

            logger.debug("Loading target unlabelled training data...")
            data_target_unlabeled = return_dataset(
                Path(self.caps_target),
                pd.read_csv(self.tsv_target_unlab, sep="\t"),
                self.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,  # A checker
                label=self.label,
                label_code=self.label_code,
            )

            logger.debug("Loading validation source data...")
            data_valid_source = return_dataset(
                self.caps_directory,
                split_df_dict["validation"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )
            logger.debug("Loading validation target labelled data...")
            data_valid_target_labeled = return_dataset(
                Path(self.caps_target),
                split_df_dict_target_lab["validation"],
                self.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,
                label=self.label,
                label_code=self.label_code,
            )
            train_source_sampler = self.task_manager.generate_sampler(
                data_train_source, self.sampler
            )

            logger.info(
                f"Getting train and validation loader with batch size {self.batch_size}"
            )

            ## Oversampling of the target dataset
            from torch.utils.data import SubsetRandomSampler

            # Create index lists for target labeled dataset
            labeled_indices = list(range(len(data_train_target_labeled)))

            # Oversample the indices for the target labeld dataset to match the size of the labeled source dataset
            data_train_source_size = len(data_train_source) // self.batch_size
            labeled_oversampled_indices = labeled_indices * (
                data_train_source_size // len(labeled_indices)
            )

            # Append remaining indices to match the size of the largest dataset
            labeled_oversampled_indices += labeled_indices[
                : data_train_source_size % len(labeled_indices)
            ]

            # Create SubsetRandomSamplers using the oversampled indices
            labeled_sampler = SubsetRandomSampler(labeled_oversampled_indices)

            train_source_loader = DataLoader(
                data_train_source,
                batch_size=self.batch_size,
                sampler=train_source_sampler,
                # shuffle=True,  # len(data_train_source) < len(data_train_target_labeled),
                num_workers=self.n_proc,
                worker_init_fn=pl_worker_init_function,
                drop_last=True,
            )
            logger.info(
                f"Train source loader size is {len(train_source_loader)*self.batch_size}"
            )
            train_target_loader = DataLoader(
                data_train_target_labeled,
                batch_size=1,  # To limit the need of oversampling
                # sampler=train_target_sampler,
                sampler=labeled_sampler,
                num_workers=self.n_proc,
                worker_init_fn=pl_worker_init_function,
                # shuffle=True,  # len(data_train_target_labeled) < len(data_train_source),
                drop_last=True,
            )
            logger.info(
                f"Train target labeled loader size oversample is {len(train_target_loader)}"
            )

            data_train_target_labeled.df = data_train_target_labeled.df[
                ["participant_id", "session_id", "diagnosis", "cohort", "domain"]
            ]

            train_target_unl_loader = DataLoader(
                data_target_unlabeled,
                batch_size=self.batch_size,
                num_workers=self.n_proc,
                # sampler=unlabeled_sampler,
                worker_init_fn=pl_worker_init_function,
                shuffle=True,
                drop_last=True,
            )

            logger.info(
                f"Train target unlabeled loader size is {len(train_target_unl_loader)*self.batch_size}"
            )

            valid_loader_source = DataLoader(
                data_valid_source,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_proc,
            )
            logger.info(
                f"Validation loader source size is {len(valid_loader_source)*self.batch_size}"
            )

            valid_loader_target = DataLoader(
                data_valid_target_labeled,
                batch_size=self.batch_size,  # To check
                shuffle=False,
                num_workers=self.n_proc,
            )
            logger.info(
                f"Validation loader target size is {len(valid_loader_target)*self.batch_size}"
            )

            self._train_ssdann(
                train_source_loader,
                train_target_loader,
                train_target_unl_loader,
                valid_loader_target,
                valid_loader_source,
                split,
                resume=resume,
            )

            self._ensemble_prediction(
                "train",
                split,
                self.selection_metrics,
            )
            self._ensemble_prediction(
                "validation",
                split,
                self.selection_metrics,
            )

            self._erase_tmp(split)

    def _train(
        self,
        train_loader,
        valid_loader,
        split,
        network=None,
        resume=False,
        callbacks=[],
    ):
        """
        Core function shared by train and resume.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader wrapping the training set.
            valid_loader (torch.utils.data.DataLoader): DataLoader wrapping the validation set.
            split (int): Index of the split trained.
            network (int): Index of the network trained (used in multi-network setting only).
            resume (bool): If True the job is resumed from the checkpoint.
        """
        self._init_callbacks()
        model, beginning_epoch = self._init_model(
            split=split,
            resume=resume,
            transfer_path=self.transfer_path,
            transfer_selection=self.transfer_selection_metric,
            nb_unfrozen_layer=self.nb_unfrozen_layer,
        )
        model = DDP(model)
        criterion = self.task_manager.get_criterion(self.loss)

        optimizer = self._init_optimizer(model, split=split, resume=resume)
        self.callback_handler.on_train_begin(
            self.parameters,
            criterion=criterion,
            optimizer=optimizer,
            split=split,
            maps_path=self.maps_path,
        )

        model.train()
        train_loader.dataset.train()

        early_stopping = EarlyStopping(
            "min", min_delta=self.tolerance, patience=self.patience
        )
        metrics_valid = {"loss": None}

        if cluster.master:
            log_writer = LogWriter(
                self.maps_path,
                self.task_manager.evaluation_metrics + ["loss"],
                split,
                resume=resume,
                beginning_epoch=beginning_epoch,
                network=network,
            )
            retain_best = RetainBest(selection_metrics=list(self.selection_metrics))
        epoch = beginning_epoch

        retain_best = RetainBest(selection_metrics=list(self.selection_metrics))

        if self.parameters["adaptive_learning_rate"]:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            # Initialize the ReduceLROnPlateau scheduler
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, verbose=True
            )

        scaler = GradScaler(enabled=self.amp)
        profiler = self._init_profiler()

        while epoch < self.epochs and not early_stopping.step(metrics_valid["loss"]):
            # self.callback_handler.on_epoch_begin(self.parameters, epoch = epoch)

            if isinstance(train_loader.sampler, DistributedSampler):
                # It should always be true for a random sampler. But just in case
                # we get a WeightedRandomSampler or a forgotten RandomSampler,
                # we do not want to execute this line.
                train_loader.sampler.set_epoch(epoch)

            model.zero_grad(set_to_none=True)
            evaluation_flag, step_flag = True, True

            with profiler:
                for i, data in enumerate(train_loader):
                    update: bool = (i + 1) % self.accumulation_steps == 0
                    sync = nullcontext() if update else model.no_sync()
                    with sync:
                        with autocast(enabled=self.amp):
                            _, loss_dict = model(data, criterion)
                        logger.debug(f"Train loss dictionnary {loss_dict}")
                        loss = loss_dict["loss"]
                        scaler.scale(loss).backward()

                    if update:
                        step_flag = False
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                        del loss

                        # Evaluate the model only when no gradients are accumulated
                        if (
                            self.evaluation_steps != 0
                            and (i + 1) % self.evaluation_steps == 0
                        ):
                            evaluation_flag = False

                            _, metrics_train = self.task_manager.test(
                                model, train_loader, criterion, amp=self.amp
                            )
                            _, metrics_valid = self.task_manager.test(
                                model, valid_loader, criterion, amp=self.amp
                            )

                            model.train()
                            train_loader.dataset.train()

                            if cluster.master:
                                log_writer.step(
                                    epoch,
                                    i,
                                    metrics_train,
                                    metrics_valid,
                                    len(train_loader),
                                )
                            logger.info(
                                f"{self.mode} level training loss is {metrics_train['loss']} "
                                f"at the end of iteration {i}"
                            )
                            logger.info(
                                f"{self.mode} level validation loss is {metrics_valid['loss']} "
                                f"at the end of iteration {i}"
                            )

                    profiler.step()

                # If no step has been performed, raise Exception
                if step_flag:
                    raise Exception(
                        "The model has not been updated once in the epoch. The accumulation step may be too large."
                    )

                # If no evaluation has been performed, warn the user
                elif evaluation_flag and self.evaluation_steps != 0:
                    logger.warning(
                        f"Your evaluation steps {self.evaluation_steps} are too big "
                        f"compared to the size of the dataset. "
                        f"The model is evaluated only once at the end epochs."
                    )

                # Update weights one last time if gradients were computed without update
                if (i + 1) % self.accumulation_steps != 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # Always test the results and save them once at the end of the epoch
                model.zero_grad(set_to_none=True)
                logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

                _, metrics_train = self.task_manager.test(
                    model, train_loader, criterion, amp=self.amp
                )
                _, metrics_valid = self.task_manager.test(
                    model, valid_loader, criterion, amp=self.amp
                )

                model.train()
                train_loader.dataset.train()

            self.callback_handler.on_epoch_end(
                self.parameters,
                metrics_train=metrics_train,
                metrics_valid=metrics_valid,
                mode=self.mode,
                i=i,
            )

            if cluster.master:
                # Save checkpoints and best models
                best_dict = retain_best.step(metrics_valid)
                self._write_weights(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "name": self.architecture,
                    },
                    best_dict,
                    split,
                    network=network,
                    save_all_models=self.parameters["save_all_models"],
                )
                self._write_weights(
                    {
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "name": self.optimizer,
                    },
                    None,
                    split,
                    filename="optimizer.pth.tar",
                    save_all_models=self.parameters["save_all_models"],
                )
            if self.parameters["adaptive_learning_rate"]:
                scheduler.step(
                    metrics_valid["loss"]
                )  # Update learning rate based on validation loss

            epoch += 1

        del model
        self._test_loader(
            train_loader,
            criterion,
            "train",
            split,
            self.selection_metrics,
            amp=self.amp,
            network=network,
        )
        self._test_loader(
            valid_loader,
            criterion,
            "validation",
            split,
            self.selection_metrics,
            amp=self.amp,
            network=network,
        )

        if self.task_manager.save_outputs:
            self._compute_output_tensors(
                train_loader.dataset,
                "train",
                split,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )
            self._compute_output_tensors(
                train_loader.dataset,
                "validation",
                split,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )

        self.callback_handler.on_train_end(parameters=self.parameters)

    def _train_ssdann(
        self,
        train_source_loader,
        train_target_loader,
        train_target_unl_loader,
        valid_loader,
        valid_source_loader,
        split,
        network=None,
        resume=False,
        evaluate_source=True,  # TO MODIFY
    ):
        """
        Core function shared by train and resume.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader wrapping the training set.
            valid_loader (torch.utils.data.DataLoader): DataLoader wrapping the validation set.
            split (int): Index of the split trained.
            network (int): Index of the network trained (used in multi-network setting only).
            resume (bool): If True the job is resumed from the checkpoint.
        """

        model, beginning_epoch = self._init_model(
            split=split,
            resume=resume,
            transfer_path=self.transfer_path,
            transfer_selection=self.transfer_selection_metric,
        )

        criterion = self.task_manager.get_criterion(self.loss)
        logger.debug(f"Criterion for {self.network_task} is {criterion}")
        optimizer = self._init_optimizer(model, split=split, resume=resume)

        logger.debug(f"Optimizer used for training is optimizer")

        model.train()
        train_source_loader.dataset.train()
        train_target_loader.dataset.train()
        train_target_unl_loader.dataset.train()

        early_stopping = EarlyStopping(
            "min", min_delta=self.tolerance, patience=self.patience
        )

        metrics_valid_target = {"loss": None}
        metrics_valid_source = {"loss": None}

        log_writer = LogWriter(
            self.maps_path,
            self.task_manager.evaluation_metrics + ["loss"],
            split,
            resume=resume,
            beginning_epoch=beginning_epoch,
            network=network,
        )
        epoch = log_writer.beginning_epoch

        retain_best = RetainBest(selection_metrics=list(self.selection_metrics))
        import numpy as np

        while epoch < self.epochs and not early_stopping.step(
            metrics_valid_target["loss"]
        ):
            logger.info(f"Beginning epoch {epoch}.")

            model.zero_grad()
            evaluation_flag, step_flag = True, True

            for i, (data_source, data_target, data_target_unl) in enumerate(
                zip(train_source_loader, train_target_loader, train_target_unl_loader)
            ):
                p = (
                    float(epoch * len(train_target_loader))
                    / 10
                    / len(train_target_loader)
                )
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                # alpha = 0
                _, _, loss_dict = model.compute_outputs_and_loss(
                    data_source, data_target, data_target_unl, criterion, alpha
                )  # TO CHECK
                logger.debug(f"Train loss dictionnary {loss_dict}")
                loss = loss_dict["loss"]
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    step_flag = False
                    optimizer.step()
                    optimizer.zero_grad()

                    del loss

                    # Evaluate the model only when no gradients are accumulated
                    if (
                        self.evaluation_steps != 0
                        and (i + 1) % self.evaluation_steps == 0
                    ):
                        evaluation_flag = False

                        # Evaluate on taget data
                        logger.info("Evaluation on target data")
                        _, metrics_train_target = self.task_manager.test_da(
                            model,
                            train_target_loader,
                            criterion,
                            alpha,
                            target=True,
                        )  # TO CHECK

                        _, metrics_valid_target = self.task_manager.test_da(
                            model,
                            valid_loader,
                            criterion,
                            alpha,
                            target=True,
                        )

                        model.train()
                        train_target_loader.dataset.train()

                        log_writer.step(
                            epoch,
                            i,
                            metrics_train_target,
                            metrics_valid_target,
                            len(train_target_loader),
                            "training_target.tsv",
                        )
                        logger.info(
                            f"{self.mode} level training loss for target data is {metrics_train_target['loss']} "
                            f"at the end of iteration {i}"
                        )
                        logger.info(
                            f"{self.mode} level validation loss for target data is {metrics_valid_target['loss']} "
                            f"at the end of iteration {i}"
                        )

                        # Evaluate on source data
                        logger.info("Evaluation on source data")
                        _, metrics_train_source = self.task_manager.test_da(
                            model, train_source_loader, criterion, alpha
                        )
                        _, metrics_valid_source = self.task_manager.test_da(
                            model, valid_source_loader, criterion, alpha
                        )

                        model.train()
                        train_source_loader.dataset.train()

                        log_writer.step(
                            epoch,
                            i,
                            metrics_train_source,
                            metrics_valid_source,
                            len(train_source_loader),
                        )
                        logger.info(
                            f"{self.mode} level training loss for source data is {metrics_train_source['loss']} "
                            f"at the end of iteration {i}"
                        )
                        logger.info(
                            f"{self.mode} level validation loss for source data is {metrics_valid_source['loss']} "
                            f"at the end of iteration {i}"
                        )

            # If no step has been performed, raise Exception
            if step_flag:
                raise Exception(
                    "The model has not been updated once in the epoch. The accumulation step may be too large."
                )

            # If no evaluation has been performed, warn the user
            elif evaluation_flag and self.evaluation_steps != 0:
                logger.warning(
                    f"Your evaluation steps {self.evaluation_steps} are too big "
                    f"compared to the size of the dataset. "
                    f"The model is evaluated only once at the end epochs."
                )

            # Update weights one last time if gradients were computed without update
            if (i + 1) % self.accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            # Always test the results and save them once at the end of the epoch
            model.zero_grad()
            logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

            if evaluate_source:
                logger.info(
                    f"Evaluate source data at the end of the epoch {epoch} with alpha: {alpha}."
                )
                _, metrics_train_source = self.task_manager.test_da(
                    model,
                    train_source_loader,
                    criterion,
                    alpha,
                    True,
                    False,
                )
                _, metrics_valid_source = self.task_manager.test_da(
                    model,
                    valid_source_loader,
                    criterion,
                    alpha,
                    True,
                    False,
                )

                log_writer.step(
                    epoch,
                    i,
                    metrics_train_source,
                    metrics_valid_source,
                    len(train_source_loader),
                )

                logger.info(
                    f"{self.mode} level training loss for source data is {metrics_train_source['loss']} "
                    f"at the end of iteration {i}"
                )
                logger.info(
                    f"{self.mode} level validation loss for source data is {metrics_valid_source['loss']} "
                    f"at the end of iteration {i}"
                )

            _, metrics_train_target = self.task_manager.test_da(
                model,
                train_target_loader,
                criterion,
                alpha,
                target=True,
            )
            _, metrics_valid_target = self.task_manager.test_da(
                model,
                valid_loader,
                criterion,
                alpha,
                target=True,
            )

            model.train()
            train_source_loader.dataset.train()
            train_target_loader.dataset.train()

            log_writer.step(
                epoch,
                i,
                metrics_train_target,
                metrics_valid_target,
                len(train_target_loader),
                "training_target.tsv",
            )

            logger.info(
                f"{self.mode} level training loss for target data is {metrics_train_target['loss']} "
                f"at the end of iteration {i}"
            )
            logger.info(
                f"{self.mode} level validation loss for target data is {metrics_valid_target['loss']} "
                f"at the end of iteration {i}"
            )

            # Save checkpoints and best models
            best_dict = retain_best.step(metrics_valid_target)
            self._write_weights(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "name": self.architecture,
                },
                best_dict,
                split,
                network=network,
                save_all_models=False,
            )
            self._write_weights(
                {
                    "optimizer": optimizer.state_dict(),  # TO MODIFY
                    "epoch": epoch,
                    "name": self.optimizer,
                },
                None,
                split,
                filename="optimizer.pth.tar",
                save_all_models=False,
            )

            epoch += 1

        self._test_loader_ssda(
            train_target_loader,
            criterion,
            data_group="train",
            split=split,
            selection_metrics=self.selection_metrics,
            network=network,
            target=True,
            alpha=0,
        )
        self._test_loader_ssda(
            valid_loader,
            criterion,
            data_group="validation",
            split=split,
            selection_metrics=self.selection_metrics,
            network=network,
            target=True,
            alpha=0,
        )

        if self.task_manager.save_outputs:
            self._compute_output_tensors(
                train_target_loader.dataset,
                "train",
                split,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )
            self._compute_output_tensors(
                train_target_loader.dataset,
                "validation",
                split,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )

    def _test_loader(
        self,
        dataloader,
        criterion,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        gpu=None,
        amp=False,
        network=None,
        report_ci=True,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            data_group (str): name of the data group used for the testing task.
            split (int): Index of the split used to train the model tested.
            selection_metrics (list[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            gpu (bool): If given, a new value for the device of the model will be computed.
            amp (bool): If enabled, uses Automatic Mixed Precision (requires GPU usage).
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            if cluster.master:
                log_dir = (
                    self.maps_path
                    / f"{self.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                )
                self.write_description_log(
                    log_dir,
                    data_group,
                    dataloader.dataset.caps_dict,
                    dataloader.dataset.df,
                )

            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            model = DDP(model)

            prediction_df, metrics = self.task_manager.test(
                model,
                dataloader,
                criterion,
                use_labels=use_labels,
                amp=amp,
                report_ci=report_ci,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{self.mode}_id"] = network

                loss_to_log = (
                    metrics["Metric_values"][-1] if report_ci else metrics["loss"]
                )

                logger.info(
                    f"{self.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            if cluster.master:
                # Replace here
                self._mode_level_to_tsv(
                    prediction_df,
                    metrics,
                    split,
                    selection_metric,
                    data_group=data_group,
                )

    def _test_loader_ssda(
        self,
        dataloader,
        criterion,
        alpha,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        gpu=None,
        network=None,
        target=False,
        report_ci=True,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            data_group (str): name of the data group used for the testing task.
            split (int): Index of the split used to train the model tested.
            selection_metrics (list[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            log_dir = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
            )
            self.write_description_log(
                log_dir,
                data_group,
                dataloader.dataset.caps_dict,
                dataloader.dataset.df,
            )

            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            prediction_df, metrics = self.task_manager.test_da(
                model, dataloader, criterion, target=target, report_ci=report_ci
            )
            if use_labels:
                if network is not None:
                    metrics[f"{self.mode}_id"] = network

                if report_ci:
                    loss_to_log = metrics["Metric_values"][-1]
                else:
                    loss_to_log = metrics["loss"]

                logger.info(
                    f"{self.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            # Replace here
            self._mode_level_to_tsv(
                prediction_df, metrics, split, selection_metric, data_group=data_group
            )

    @torch.no_grad()
    def _compute_output_nifti(
        self,
        dataset,
        data_group,
        split,
        selection_metrics,
        gpu=None,
        network=None,
    ):
        """
        Computes the output nifti images and saves them in the MAPS.

        Args:
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        # Raise an error if mode is not image
        """
        import nibabel as nib
        from numpy import eye

        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=self.nb_unfrozen_layer,
            )
            model = DDP(model)

            nifti_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
                / "nifti_images"
            )
            if cluster.master:
                nifti_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            nb_imgs = len(dataset)
            for i in range(cluster.rank, nb_imgs, cluster.world_size):
                data = dataset[i]
                image = data["image"]
                x = image.unsqueeze(0).to(model.device)
                with autocast(enabled=self.amp):
                    output = model.predict(x)
                output = output.squeeze(0).detach().cpu().float()
                # Convert tensor to nifti image with appropriate affine
                input_nii = nib.Nifti1Image(image[0].detach().cpu().numpy(), eye(4))
                output_nii = nib.Nifti1Image(output[0].numpy(), eye(4))
                # Create file name according to participant and session id
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                input_filename = f"{participant_id}_{session_id}_image_input.nii.gz"
                output_filename = f"{participant_id}_{session_id}_image_output.nii.gz"
                nib.save(input_nii, nifti_path / input_filename)
                nib.save(output_nii, nifti_path / output_filename)

    @torch.no_grad()
    def _compute_output_tensors(
        self,
        dataset,
        data_group,
        split,
        selection_metrics,
        nb_images=None,
        gpu=None,
        network=None,
    ):
        """
        Compute the output tensors and saves them in the MAPS.

        Args:
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            nb_images (int): number of full images to write. Default computes the outputs of the whole data set.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=self.nb_unfrozen_layer,
            )
            model = DDP(model)

            tensor_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
                / "tensors"
            )
            if cluster.master:
                tensor_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            if nb_images is None:  # Compute outputs for the whole data set
                nb_modes = len(dataset)
            else:
                nb_modes = nb_images * dataset.elem_per_image

            for i in range(cluster.rank, nb_modes, cluster.world_size):
                data = dataset[i]
                image = data["image"]
                x = image.unsqueeze(0).to(model.device)
                with autocast(enabled=self.amp):
                    output = model.predict(x)
                output = output.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{self.mode}_id"]
                input_filename = (
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_input.pt"
                )
                output_filename = (
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_output.pt"
                )
                torch.save(image, tensor_path / input_filename)
                torch.save(output, tensor_path / output_filename)
                logger.debug(f"File saved at {[input_filename, output_filename]}")

    def _compute_latent_tensors(
        self,
        dataset,
        data_group,
        split,
        selection_metrics,
        nb_images=None,
        gpu=None,
        network=None,
    ):
        """
        Compute the output tensors and saves them in the MAPS.

        Args:
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            split (int): split number.
            selection_metrics (list[str]): metrics used for model selection.
            nb_images (int): number of full images to write. Default computes the outputs of the whole data set.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=self.nb_unfrozen_layer,
            )
            model = DDP(model)

            tensor_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / f"best-{selection_metric}"
                / data_group
                / "latent_tensors"
            )
            if cluster.master:
                tensor_path.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            if nb_images is None:  # Compute outputs for the whole data set
                nb_modes = len(dataset)
            else:
                nb_modes = nb_images * dataset.elem_per_image

            for i in range(cluster.rank, nb_modes, cluster.world_size):
                data = dataset[i]
                image = data["image"]
                logger.debug(f"Image for latent representation {image}")
                with autocast(enabled=self.amp):
                    _, latent, _ = model._forward(image.unsqueeze(0).to(model.device))
                latent = latent.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{self.mode}_id"]
                output_filename = (
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_latent.pt"
                )
                torch.save(latent, tensor_path / output_filename)

    def _ensemble_prediction(
        self,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        skip_leak_check=False,
    ):
        """Computes the results on the image-level."""

        if not selection_metrics:
            selection_metrics = self._find_selection_metrics(split)

        for selection_metric in selection_metrics:
            #####################
            # Soft voting
            if self.num_networks > 1 and not skip_leak_check:
                self._ensemble_to_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
            elif self.mode != "image" and not skip_leak_check:
                self._mode_to_image_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )

    ###############################
    # Checks                      #
    ###############################
    def _check_args(self, parameters):
        """
        Check the training parameters integrity
        """
        logger.debug("Checking arguments...")
        mandatory_arguments = [
            "caps_directory",
            "tsv_path",
            "preprocessing_dict",
            "mode",
            "network_task",
        ]
        for arg in mandatory_arguments:
            if arg not in parameters:
                raise ClinicaDLArgumentError(
                    f"The values of mandatory arguments {mandatory_arguments} should be set. "
                    f"No value was given for {arg}."
                )
        self.parameters = add_default_values(parameters)
        self.parameters = change_str_to_path(parameters)
        if self.parameters["gpu"]:
            check_gpu()
        elif self.parameters["amp"]:
            raise ClinicaDLArgumentError(
                "AMP is designed to work with modern GPUs. Please add the --gpu flag."
            )

        _, transformations = get_transforms(
            normalize=self.normalize,
            size_reduction=self.size_reduction,
            size_reduction_factor=self.size_reduction_factor,
        )

        split_manager = self._init_split_manager(None)
        train_df = split_manager[0]["train"]
        if "label" not in self.parameters:
            self.parameters["label"] = None

        self.task_manager = self._init_task_manager(df=train_df)

        if self.parameters["architecture"] == "default":
            self.parameters["architecture"] = self.task_manager.get_default_network()
        if "selection_threshold" not in self.parameters:
            self.parameters["selection_threshold"] = None
        if (
            "label_code" not in self.parameters
            or len(self.parameters["label_code"]) == 0
        ):  # Allows to set custom label code in TOML
            self.parameters["label_code"] = self.task_manager.generate_label_code(
                train_df, self.label
            )

        full_dataset = return_dataset(
            self.caps_directory,
            train_df,
            self.preprocessing_dict,
            multi_cohort=self.multi_cohort,
            label=self.label,
            label_code=self.parameters["label_code"],
            train_transformations=None,
            all_transformations=transformations,
        )
        self.parameters.update(
            {
                "num_networks": full_dataset.elem_per_image,
                "output_size": self.task_manager.output_size(
                    full_dataset.size, full_dataset.df, self.label
                ),
                "input_size": full_dataset.size,
            }
        )

        self.parameters["seed"] = get_seed(self.parameters["seed"])

        if self.parameters["num_networks"] < 2 and self.multi_network:
            raise ClinicaDLConfigurationError(
                f"Invalid training configuration: cannot train a multi-network "
                f"framework with only {self.parameters['num_networks']} element "
                f"per image."
            )
        possible_selection_metrics_set = set(self.task_manager.evaluation_metrics) | {
            "loss"
        }
        if not set(self.parameters["selection_metrics"]).issubset(
            possible_selection_metrics_set
        ):
            raise ClinicaDLConfigurationError(
                f"Selection metrics {self.parameters['selection_metrics']} "
                f"must be a subset of metrics used for evaluation "
                f"{possible_selection_metrics_set}."
            )

    def _check_split_wording(self):
        """Finds if MAPS structure uses 'fold-X' or 'split-X' folders."""

        if len(list(self.maps_path.glob("fold-*"))) > 0:
            return "fold"
        else:
            return "split"

    def _find_splits(self):
        """Find which splits were trained in the MAPS."""
        return [
            int(split.name.split("-")[1])
            for split in list(self.maps_path.iterdir())
            if split.name.startswith(f"{self.split_name}-")
        ]

    def _find_selection_metrics(self, split):
        """Find which selection metrics are available in MAPS for a given split."""

        split_path = self.maps_path / f"{self.split_name}-{split}"
        if not split_path.is_dir():
            raise MAPSError(
                f"Training of split {split} was not performed."
                f"Please execute maps_manager.train(split_list=[{split}])"
            )

        return [
            metric.name.split("-")[1]
            for metric in list(split_path.iterdir())
            if metric.name[:5:] == "best-"
        ]

    def _check_selection_metric(self, split, selection_metric=None):
        """Check that a given selection metric is available for a given split."""
        available_metrics = self._find_selection_metrics(split)
        if not selection_metric:
            if len(available_metrics) > 1:
                raise ClinicaDLArgumentError(
                    f"Several metrics are available for split {split}. "
                    f"Please choose which one you want to read among {available_metrics}"
                )
            else:
                selection_metric = available_metrics[0]
        else:
            if selection_metric not in available_metrics:
                raise ClinicaDLArgumentError(
                    f"The metric {selection_metric} is not available."
                    f"Please choose among is the available metrics {available_metrics}."
                )
        return selection_metric

    def _check_leakage(self, data_group, test_df):
        """
        Checks that no intersection exist between the participants used for training and those used for testing.

        Args:
            data_group (str): name of the data group
            test_df (pd.DataFrame): Table of participant_id / session_id of the data group
        Raises:
            ClinicaDLDataLeakageError: if data_group not in ["train", "validation"] and there is an intersection
                between the participant IDs in test_df and the ones used for training.
        """
        if data_group not in ["train", "validation"]:
            train_path = self.maps_path / "groups" / "train+validation.tsv"
            train_df = pd.read_csv(train_path, sep="\t")
            participants_train = set(train_df.participant_id.values)
            participants_test = set(test_df.participant_id.values)
            intersection = participants_test & participants_train

            if len(intersection) > 0:
                raise ClinicaDLDataLeakageError(
                    "Your evaluation set contains participants who were already seen during "
                    "the training step. The list of common participants is the following: "
                    f"{intersection}."
                )

    def _check_data_group(
        self,
        data_group,
        caps_directory=None,
        df=None,
        multi_cohort=False,
        overwrite=False,
        label=None,
        split_list=None,
        skip_leak_check=False,
    ):
        """
        Check if a data group is already available if other arguments are None.
        Else creates a new data_group.

        Args:
            data_group (str): name of the data group
            caps_directory  (str): input CAPS directory
            df (pd.DataFrame): Table of participant_id / session_id of the data group
            multi_cohort (bool): indicates if the input data comes from several CAPS
            overwrite (bool): If True former definition of data group is erased
            label (str): label name if applicable

        Raises:
            MAPSError when trying to overwrite train or validation data groups
            ClinicaDLArgumentError:
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
        """
        group_dir = self.maps_path / "groups" / data_group
        logger.debug(f"Group path {group_dir}")
        if group_dir.is_dir():  # Data group already exists
            if overwrite:
                if data_group in ["train", "validation"]:
                    raise MAPSError("Cannot overwrite train or validation data group.")
                else:
                    if not split_list:
                        split_list = self._find_splits()
                    for split in split_list:
                        selection_metrics = self._find_selection_metrics(split)
                        for selection in selection_metrics:
                            results_path = (
                                self.maps_path
                                / f"{self.split_name}-{split}"
                                / f"best-{selection}"
                                / data_group
                            )
                            if results_path.is_dir():
                                shutil.rmtree(results_path)
            elif df is not None or caps_directory is not None:
                raise ClinicaDLArgumentError(
                    f"Data group {data_group} is already defined. "
                    f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                    f"To erase {data_group} please set overwrite to True."
                )

        elif not group_dir.is_dir() and (
            caps_directory is None or df is None
        ):  # Data group does not exist yet / was overwritten + missing data
            raise ClinicaDLArgumentError(
                f"The data group {data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        elif (
            not group_dir.is_dir()
        ):  # Data group does not exist yet / was overwritten + all data is provided
            if skip_leak_check:
                logger.info("Skipping data leakage check")
            else:
                self._check_leakage(data_group, df)
            self._write_data_group(
                data_group, df, caps_directory, multi_cohort, label=label
            )

    ###############################
    # File writers                #
    ###############################
    @staticmethod
    def write_parameters(json_path: Path, parameters, verbose=True):
        """Write JSON files of parameters."""
        logger.debug("Writing parameters...")
        json_path.mkdir(parents=True, exist_ok=True)

        parameters = change_path_to_str(parameters)
        # save to json file
        json_data = json.dumps(parameters, skipkeys=True, indent=4)
        json_path = json_path / "maps.json"
        if verbose:
            logger.info(f"Path of json file: {json_path}")
        with json_path.open(mode="w") as f:
            f.write(json_data)
        parameters = change_str_to_path(parameters)

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        logger.debug("Writing requirement version...")
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with (self.maps_path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_training_data(self):
        """Writes the TSV file containing the participant and session IDs used for training."""
        logger.debug("Writing training data...")
        from clinicadl.utils.caps_dataset.data import load_data_test

        train_df = load_data_test(
            self.tsv_path,
            self.diagnoses,
            baseline=False,
            multi_cohort=self.multi_cohort,
        )
        train_df = train_df[["participant_id", "session_id"]]
        if self.transfer_path:
            transfer_train_path = self.transfer_path / "groups" / "train+validation.tsv"
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(
            self.maps_path / "groups" / "train+validation.tsv", sep="\t", index=False
        )

    def _write_data_group(
        self,
        data_group,
        df,
        caps_directory: Path = None,
        multi_cohort: bool = None,
        label=None,
    ):
        """
        Check that a data_group is not already written and writes the characteristics of the data group
        (TSV file with a list of participant / session + JSON file containing the CAPS and the preprocessing).

        Args:
            data_group (str): name whose presence is checked.
            df (pd.DataFrame): DataFrame containing the participant_id and session_id (and label if use_labels is True)
            caps_directory (str): caps_directory if different from the training caps_directory,
            multi_cohort (bool): multi_cohort used if different from the training multi_cohort.
        """
        group_path = self.maps_path / "groups" / data_group
        group_path.mkdir(parents=True)

        columns = ["participant_id", "session_id", "cohort"]
        if self.label in df.columns.values:
            columns += [self.label]
        if label is not None and label in df.columns.values:
            columns += [label]

        df.to_csv(group_path / "data.tsv", sep="\t", columns=columns, index=False)
        self.write_parameters(
            group_path,
            {
                "caps_directory": (
                    caps_directory
                    if caps_directory is not None
                    else self.caps_directory
                ),
                "multi_cohort": (
                    multi_cohort if multi_cohort is not None else self.multi_cohort
                ),
            },
        )

    def _write_train_val_groups(self):
        """Defines the training and validation groups at the initialization"""
        logger.debug("Writing training and validation groups...")
        split_manager = self._init_split_manager()
        for split in split_manager.split_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[split][data_group]
                group_path = (
                    self.maps_path
                    / "groups"
                    / data_group
                    / f"{self.split_name}-{split}"
                )
                group_path.mkdir(parents=True, exist_ok=True)

                columns = ["participant_id", "session_id", "cohort"]
                if self.label is not None:
                    columns.append(self.label)
                df.to_csv(group_path / "data.tsv", sep="\t", columns=columns)
                self.write_parameters(
                    group_path,
                    {
                        "caps_directory": self.caps_directory,
                        "multi_cohort": self.multi_cohort,
                    },
                    verbose=False,
                )

    def _write_weights(
        self,
        state: Dict[str, Any],
        metrics_dict: Optional[Dict[str, bool]],
        split: int,
        network: int = None,
        filename: str = "checkpoint.pth.tar",
        save_all_models: bool = False,
    ):
        """
        Update checkpoint and save the best model according to a set of metrics.
        If no metrics_dict is given, only the checkpoint is saved.

        Args:
            state: state of the training (model weights, epoch...).
            metrics_dict: output of RetainBest step.
            split: split number.
            network: network number (multi-network framework).
            filename: name of the checkpoint file.
        """
        checkpoint_dir = self.maps_path / f"{self.split_name}-{split}" / "tmp"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        torch.save(state, checkpoint_path)

        if save_all_models:
            all_models_dir = (
                self.maps_path / f"{self.split_name}-{split}" / "all_models"
            )
            all_models_dir.mkdir(parents=True, exist_ok=True)
            torch.save(state, all_models_dir / f"model_epoch_{state['epoch']}.pth.tar")

        best_filename = "model.pth.tar"
        if network is not None:
            best_filename = f"network-{network}_model.pth.tar"

        # Save model according to several metrics
        if metrics_dict is not None:
            for metric_name, metric_bool in metrics_dict.items():
                metric_path = (
                    self.maps_path
                    / f"{self.split_name}-{split}"
                    / f"best-{metric_name}"
                )
                if metric_bool:
                    metric_path.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(checkpoint_path, metric_path / best_filename)

    def _write_information(self):
        """
        Writes model architecture of the MAPS in MAPS root.
        """
        from datetime import datetime

        import clinicadl.utils.network as network_package

        model_class = getattr(network_package, self.architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = self.parameters[arg]
        kwargs["gpu"] = False

        model = model_class(**kwargs)

        file_name = "information.log"

        with (self.maps_path / file_name).open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.maps_path}\n\n")
            # f.write("- Job ID :\t{}\n".format(os.getenv('SLURM_JOBID')))
            f.write(f"- Model :\t{model.layers}\n\n")

        del model

    def _erase_tmp(self, split):
        """Erase checkpoints of the model and optimizer at the end of training."""
        tmp_path = self.maps_path / f"{self.split_name}-{split}" / "tmp"
        shutil.rmtree(tmp_path)

    @staticmethod
    def write_description_log(
        log_dir: Path,
        data_group,
        caps_dict,
        df,
    ):
        """
        Write description log file associated to a data group.

        Args:
            log_dir (str): path to the log file directory.
            data_group (str): name of the data group used for the task.
            caps_dict (dict[str, str]): Dictionary of the CAPS folders used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
        """
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "description.log"
        with log_path.open(mode="w") as f:
            f.write(f"Prediction {data_group} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dict}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")

    def _mode_level_to_tsv(
        self,
        results_df: pd.DataFrame,
        metrics: Union[Dict, pd.DataFrame],
        split: int,
        selection: str,
        data_group: str = "train",
    ):
        """
        Writes the outputs of the test function in tsv files.

        Args:
            results_df: the individual results per patch.
            metrics: the performances obtained on a series of metrics.
            split: the split for which the performances were obtained.
            selection: the metrics on which the model was selected (BA, loss...)
            data_group: the name referring to the data group on which evaluation is performed.
        """
        performance_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection}"
            / data_group
        )
        performance_dir.mkdir(parents=True, exist_ok=True)
        performance_path = (
            performance_dir / f"{data_group}_{self.mode}_level_prediction.tsv"
        )
        if not performance_path.is_file():
            results_df.to_csv(performance_path, index=False, sep="\t")
        else:
            results_df.to_csv(
                performance_path, index=False, sep="\t", mode="a", header=False
            )

        metrics_path = performance_dir / f"{data_group}_{self.mode}_level_metrics.tsv"
        if metrics is not None:
            # if data_group == "train" or data_group == "validation":
            #     pd_metrics = pd.DataFrame(metrics, index = [0])
            #     header = True
            # else:
            #     pd_metrics = pd.DataFrame(metrics).T
            #     header = False

            pd_metrics = pd.DataFrame(metrics).T
            header = False
            # import ipdb; ipdb.set_trace()
            if not metrics_path.is_file():
                pd_metrics.to_csv(metrics_path, index=False, sep="\t", header=header)
            else:
                pd_metrics.to_csv(
                    metrics_path, index=False, sep="\t", mode="a", header=header
                )

    def _ensemble_to_tsv(
        self,
        split: int,
        selection: str,
        data_group: str = "test",
        use_labels: bool = True,
    ):
        """
        Writes image-level performance files from mode level performances.

        Args:
            split: split number of the cross-validation.
            selection: metric on which the model is selected (for example loss or BA).
            data_group: the name referring to the data group on which evaluation is performed.
                If different from training or validation, the weights of soft voting will be computed
                on validation accuracies.
            use_labels: If True the labels are added to the final tsv
        """
        # Choose which dataset is used to compute the weights of soft voting.
        if data_group in ["train", "validation"]:
            validation_dataset = data_group
        else:
            validation_dataset = "validation"
        test_df = self.get_prediction(
            data_group, split, selection, self.mode, verbose=False
        )
        validation_df = self.get_prediction(
            validation_dataset, split, selection, self.mode, verbose=False
        )

        performance_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection}"
            / data_group
        )

        performance_dir.mkdir(parents=True, exist_ok=True)

        df_final, metrics = self.task_manager.ensemble_prediction(
            test_df,
            validation_df,
            selection_threshold=self.selection_threshold,
            use_labels=use_labels,
        )

        if df_final is not None:
            df_final.to_csv(
                performance_dir / f"{data_group}_image_level_prediction.tsv",
                index=False,
                sep="\t",
            )
        if metrics is not None:
            pd.DataFrame(metrics, index=[0]).to_csv(
                performance_dir / f"{data_group}_image_level_metrics.tsv",
                index=False,
                sep="\t",
            )

    def _mode_to_image_tsv(
        self,
        split: int,
        selection: str,
        data_group: str = "test",
        use_labels: bool = True,
    ):
        """
        Copy mode-level TSV files to name them as image-level TSV files

        Args:
            split: split number of the cross-validation.
            selection: metric on which the model is selected (for example loss or BA)
            data_group: the name referring to the data group on which evaluation is performed.
            use_labels: If True the labels are added to the final tsv

        """
        sub_df = self.get_prediction(
            data_group, split, selection, self.mode, verbose=False
        )
        sub_df.rename(columns={f"{self.mode}_id": "image_id"}, inplace=True)

        performance_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection}"
            / data_group
        )
        sub_df.to_csv(
            performance_dir / f"{data_group}_image_level_prediction.tsv",
            index=False,
            sep="\t",
        )
        if use_labels:
            metrics_df = pd.read_csv(
                performance_dir / f"{data_group}_{self.mode}_level_metrics.tsv",
                sep="\t",
            )
            if f"{self.mode}_id" in metrics_df:
                del metrics_df[f"{self.mode}_id"]
            metrics_df.to_csv(
                (performance_dir / f"{data_group}_image_level_metrics.tsv"),
                index=False,
                sep="\t",
            )

    ###############################
    # Objects initialization      #
    ###############################
    def _init_model(
        self,
        transfer_path: Path = None,
        transfer_selection=None,
        nb_unfrozen_layer=0,
        split=None,
        resume=False,
        gpu=None,
        network=None,
    ):
        """
        Instantiate the model

        Args:
            transfer_path (str): path to a MAPS in which a model's weights are used for transfer learning.
            transfer_selection (str): name of the metric used to find the source model.
            split (int): Index of the split (only used if transfer_path is not None of not resume).
            resume (bool): If True initialize the network with the checkpoint weights.
            gpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network trained (used in multi-network setting only).
        """
        import clinicadl.utils.network as network_package

        logger.debug(f"Initialization of model {self.architecture}")
        # or choose to implement a dictionary
        model_class = getattr(network_package, self.architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = self.parameters[arg]

        # Change device from the training parameters
        if gpu is not None:
            kwargs["gpu"] = gpu

        model = model_class(**kwargs)
        logger.debug(f"Model:\n{model.layers}")

        device = "cpu"
        if device != model.device:
            device = model.device
            logger.info(f"Working on {device}")
        current_epoch = 0

        if resume:
            checkpoint_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / "tmp"
                / "checkpoint.pth.tar"
            )
            checkpoint_state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint_state["model"])
            current_epoch = checkpoint_state["epoch"]
        elif transfer_path:
            logger.debug(f"Transfer weights from MAPS at {transfer_path}")
            transfer_maps = MapsManager(transfer_path)
            transfer_state = transfer_maps.get_state_dict(
                split,
                selection_metric=transfer_selection,
                network=network,
                map_location=model.device,
            )
            transfer_class = getattr(network_package, transfer_maps.architecture)
            logger.debug(f"Transfer from {transfer_class}")
            model.transfer_weights(transfer_state["model"], transfer_class)

            if nb_unfrozen_layer != 0:
                list_name = [name for (name, _) in model.named_parameters()]
                list_param = [param for (_, param) in model.named_parameters()]

                for param, _ in zip(list_param, list_name):
                    param.requires_grad = False

                for i in range(nb_unfrozen_layer * 2):  # Unfreeze the last layers
                    param = list_param[len(list_param) - i - 1]
                    name = list_name[len(list_name) - i - 1]
                    param.requires_grad = True
                    logger.info(f"Layer {name} unfrozen {param.requires_grad}")

        return model, current_epoch

    def _init_optimizer(self, model, split=None, resume=False):
        """Initialize the optimizer and use checkpoint weights if resume is True."""

        optimizer_cls = getattr(torch.optim, self.optimizer)
        parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer_kwargs = dict(
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.fully_sharded_data_parallel:
            optimizer = optimizer_cls(parameters, **optimizer_kwargs)
        else:
            from torch.distributed.optim import ZeroRedundancyOptimizer

            optimizer = ZeroRedundancyOptimizer(
                parameters, optimizer_class=optimizer_cls, **optimizer_kwargs
            )

        if resume:
            checkpoint_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / "tmp"
                / "optimizer.pth.tar"
            )
            checkpoint_state = torch.load(checkpoint_path, map_location=model.device)
            optimizer.load_state_dict(checkpoint_state["optimizer"])

        return optimizer

    def _init_split_manager(self, split_list=None):
        from clinicadl.utils import split_manager

        split_class = getattr(split_manager, self.validation)
        args = list(
            split_class.__init__.__code__.co_varnames[
                : split_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        args.remove("split_list")
        kwargs = {"split_list": split_list}
        for arg in args:
            kwargs[arg] = self.parameters[arg]
        return split_class(**kwargs)

    def _init_split_manager_ssda(self, caps_dir, tsv_dir, split_list=None):
        # A intégrer directement dans _init_split_manager
        from clinicadl.utils import split_manager

        split_class = getattr(split_manager, self.validation)
        args = list(
            split_class.__init__.__code__.co_varnames[
                : split_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        args.remove("split_list")
        kwargs = {"split_list": split_list}
        for arg in args:
            kwargs[arg] = self.parameters[arg]

        kwargs["caps_directory"] = Path(caps_dir)
        kwargs["tsv_path"] = Path(tsv_dir)

        return split_class(**kwargs)

    def _init_task_manager(self, df=None, n_classes=None):
        from clinicadl.utils.task_manager import (
            ClassificationManager,
            ReconstructionManager,
            RegressionManager,
        )

        if self.network_task == "classification":
            if n_classes is not None:
                return ClassificationManager(self.mode, n_classes=n_classes)
            else:
                return ClassificationManager(self.mode, df=df, label=self.label)
        elif self.network_task == "regression":
            return RegressionManager(self.mode)
        elif self.network_task == "reconstruction":
            return ReconstructionManager(self.mode)
        else:
            raise NotImplementedError(
                f"Task {self.network_task} is not implemented in ClinicaDL. "
                f"Please choose between classification, regression and reconstruction."
            )

    def _init_profiler(self):
        if self.profiler:
            from clinicadl.utils.maps_manager.cluster.profiler import (
                ProfilerActivity,
                profile,
                schedule,
                tensorboard_trace_handler,
            )

            time = datetime.now().strftime("%H:%M:%S")
            filename = [self.maps_path / "profiler" / f"clinicadl_{time}"]
            dist.broadcast_object_list(filename, src=0)
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=2, warmup=2, active=30, repeat=1),
                on_trace_ready=tensorboard_trace_handler(filename[0]),
                profile_memory=True,
                record_shapes=False,
                with_stack=False,
                with_flops=False,
            )
        else:
            profiler = nullcontext()
            profiler.step = lambda *args, **kwargs: None
        return profiler

    ###############################
    # Getters                     #
    ###############################
    def _print_description_log(
        self,
        data_group,
        split,
        selection_metric,
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            data_group (str): name of the data group used for the task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
        """
        log_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection_metric}"
            / data_group
        )
        log_path = log_dir / "description.log"
        with log_path.open(mode="r") as f:
            content = f.read()

    def get_group_info(
        self, data_group: str, split: int = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Gets information from corresponding data group
        (list of participant_id / session_id + configuration parameters).
        split is only needed if data_group is train or validation.
        """
        group_path = self.maps_path / "groups" / data_group
        if not group_path.is_dir():
            raise MAPSError(
                f"Data group {data_group} is not defined. "
                f"Please run a prediction to create this data group."
            )
        if data_group in ["train", "validation"]:
            if split is None:
                raise MAPSError(
                    f"Information on train or validation data can only be "
                    f"loaded if a split number is given"
                )
            elif not (group_path / f"{self.split_name}-{split}").is_dir():
                raise MAPSError(
                    f"Split {split} is not available for data group {data_group}."
                )
            else:
                group_path = group_path / f"{self.split_name}-{split}"

        df = pd.read_csv(group_path / "data.tsv", sep="\t")
        json_path = group_path / "maps.json"
        with json_path.open(mode="r") as f:
            parameters = json.load(f)
        parameters = change_str_to_path(parameters)
        return df, parameters

    def get_parameters(self):
        """Returns the training parameters dictionary."""
        json_path = self.maps_path / "maps.json"
        return read_json(json_path)

    # def get_model(
    #     self, split: int = 0, selection_metric: str = None, network: int = None
    # ) -> Network:
    #     selection_metric = self._check_selection_metric(split, selection_metric)
    #     if self.multi_network:
    #         if network is None:
    #             raise ClinicaDLArgumentError(
    #                 "Please precise the network number that must be loaded."
    #             )
    #     return self._init_model(
    #         self.maps_path,
    #         selection_metric,
    #         split,
    #         network=network,
    #         nb_unfrozen_layer=self.nb_unfrozen_layer,
    #     )[0]

    # def get_best_epoch(
    #     self, split: int = 0, selection_metric: str = None, network: int = None
    # ) -> int:
    #     selection_metric = self._check_selection_metric(split, selection_metric)
    #     if self.multi_network:
    #         if network is None:
    #             raise ClinicaDLArgumentError(
    #                 "Please precise the network number that must be loaded."
    #             )
    #     return self.get_state_dict(split=split, selection_metric=selection_metric)[
    #         "epoch"
    #     ]

    def get_state_dict(
        self, split=0, selection_metric=None, network=None, map_location=None
    ):
        """
        Get the model trained corresponding to one split and one metric evaluated on the validation set.

        Args:
            split (int): Index of the split used for training.
            selection_metric (str): name of the metric used for the selection.
            network (int): Index of the network trained (used in multi-network setting only).
            map_location (str): torch.device object or a string containing a device tag,
                it indicates the location where all tensors should be loaded.
                (see https://pytorch.org/docs/stable/generated/torch.load.html).
        Returns:
            (Dict): dictionary of results (weights, epoch number, metrics values)
        """
        selection_metric = self._check_selection_metric(split, selection_metric)
        if self.multi_network:
            if network is None:
                raise ClinicaDLArgumentError(
                    "Please precise the network number that must be loaded."
                )
            else:
                model_path = (
                    self.maps_path
                    / f"{self.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / f"network-{network}_model.pth.tar"
                )
        else:
            model_path = (
                self.maps_path
                / f"{self.split_name}-{split}"
                / f"best-{selection_metric}"
                / "model.pth.tar"
            )

        logger.info(
            f"Loading model trained for split {split} "
            f"selected according to best validation {selection_metric} "
            f"at path {model_path}."
        )
        return torch.load(model_path, map_location=map_location)

    def get_prediction(
        self, data_group, split=0, selection_metric=None, mode="image", verbose=False
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
        selection_metric = self._check_selection_metric(split, selection_metric)
        if verbose:
            self._print_description_log(data_group, split, selection_metric)
        prediction_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection_metric}"
            / data_group
        )
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

    def get_metrics(
        self, data_group, split=0, selection_metric=None, mode="image", verbose=True
    ):
        """
        Get the metrics corresponding to a group of participants identified by its data_group.

        Args:
            data_group (str): name of the data group used for the prediction task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
            mode (str): level of the prediction
            verbose (bool): if True will print associated prediction.log
        Returns:
            (dict[str:float]): Values of the metrics
        """
        selection_metric = self._check_selection_metric(split, selection_metric)
        if verbose:
            self._print_description_log(data_group, split, selection_metric)
        prediction_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection_metric}"
            / data_group
        )
        if not prediction_dir.is_dir():
            raise MAPSError(
                f"No prediction corresponding to data group {data_group} was found."
            )
        df = pd.read_csv(
            prediction_dir / f"{data_group}_{mode}_level_metrics.tsv", sep="\t"
        )
        return df.to_dict("records")[0]

    def get_interpretation(
        self,
        data_group,
        name,
        split=0,
        selection_metric=None,
        verbose=True,
        participant_id=None,
        session_id=None,
        mode_id=0,
    ) -> torch.Tensor:
        """
        Get the individual interpretation maps for one session if participant_id and session_id are filled.
        Else load the mean interpretation map.

        Args:
            data_group (str): Name of the data group used for the interpretation task.
            name (str): name of the interpretation task.
            split (int): Index of the split used for training.
            selection_metric (str): Metric used for best weights selection.
            verbose (bool): if True will print associated prediction.log.
            participant_id (str): ID of the participant (if not given load mean map).
            session_id (str): ID of the session (if not give load the mean map).
            mode_id (int): Index of the mode used.
        Returns:
            (torch.Tensor): Tensor of the interpretability map.
        """

        selection_metric = self._check_selection_metric(split, selection_metric)
        if verbose:
            self._print_description_log(data_group, split, selection_metric)
        map_dir = (
            self.maps_path
            / f"{self.split_name}-{split}"
            / f"best-{selection_metric}"
            / data_group
            / f"interpret-{name}"
        )
        if not map_dir.is_dir():
            raise MAPSError(
                f"No prediction corresponding to data group {data_group} and "
                f"interpretation {name} was found."
            )
        if participant_id is None and session_id is None:
            map_pt = torch.load(map_dir / f"mean_{self.mode}-{mode_id}_map.pt")
        elif participant_id is None or session_id is None:
            raise ValueError(
                f"To load the mean interpretation map, "
                f"please do not give any participant_id or session_id.\n "
                f"Else specify both parameters"
            )
        else:
            map_pt = torch.load(
                map_dir / f"{participant_id}_{session_id}_{self.mode}-{mode_id}_map.pt"
            )
        return map_pt

    def _init_callbacks(self):
        from clinicadl.utils.callbacks.callbacks import (
            Callback,
            CallbacksHandler,
            LoggerCallback,
        )

        # if self.callbacks is None:
        #     self.callbacks = [Callback()]

        self.callback_handler = CallbacksHandler()  # callbacks=self.callbacks)

        if self.parameters["emissions_calculator"]:
            from clinicadl.utils.callbacks.callbacks import CodeCarbonTracker

            self.callback_handler.add_callback(CodeCarbonTracker())

        if self.parameters["track_exp"]:
            from clinicadl.utils.callbacks.callbacks import Tracker

            self.callback_handler.add_callback(Tracker)

        self.callback_handler.add_callback(LoggerCallback())
        # self.callback_handler.add_callback(MetricConsolePrinterCallback())
