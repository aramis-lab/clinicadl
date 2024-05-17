from __future__ import annotations  # noqa: I001

import shutil
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.caps_dataset.data import return_dataset
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.exceptions import MAPSError
from clinicadl.utils.maps_manager.ddp import DDP, cluster
from clinicadl.utils.maps_manager.logwriter import LogWriter
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.seed import pl_worker_init_function, seed_everything
from clinicadl.utils.transforms.transforms import get_transforms

from clinicadl.utils.maps_manager import MapsManager
from clinicadl.utils.seed import get_seed

if TYPE_CHECKING:
    from clinicadl.utils.callbacks.callbacks import Callback

    from .training_config import TrainingConfig

logger = getLogger("clinicadl.trainer")


class Trainer:
    """Temporary Trainer extracted from the MAPSManager."""

    def __init__(
        self,
        config: TrainingConfig,
    ) -> None:
        """
        Parameters
        ----------
        config : BaseTaskConfig
        """
        self.config = config
        self.maps_manager = self._init_maps_manager(config)
        self._check_args()

    def _init_maps_manager(self, config) -> MapsManager:
        # temporary: to match CLI data. TODO : change CLI data
        parameters = {}
        config_dict = config.model_dump()
        for key in config_dict:
            if isinstance(config_dict[key], dict):
                parameters.update(config_dict[key])
            else:
                parameters[key] = config_dict[key]

        maps_path = parameters["output_maps_directory"]
        del parameters["output_maps_directory"]
        for parameter in parameters:
            if parameters[parameter] == Path("."):
                parameters[parameter] = ""
        if parameters["transfer_path"] is None:
            parameters["transfer_path"] = False
        if parameters["data_augmentation"] == ():
            parameters["data_augmentation"] = False
        parameters["preprocessing_dict_target"] = parameters[
            "preprocessing_json_target"
        ]
        del parameters["preprocessing_json_target"]
        del parameters["preprocessing_json"]
        parameters["tsv_path"] = parameters["tsv_directory"]
        del parameters["tsv_directory"]
        parameters["compensation"] = parameters["compensation"].value
        parameters["size_reduction_factor"] = parameters["size_reduction_factor"].value
        if parameters["track_exp"]:
            parameters["track_exp"] = parameters["track_exp"].value
        else:
            parameters["track_exp"] = ""
        parameters["sampler"] = parameters["sampler"].value
        if parameters["network_task"] == "reconstruction":
            parameters["normalization"] = parameters["normalization"].value
        ###############################
        return MapsManager(
            maps_path, parameters, verbose=None
        )  # TODO : precise which parameters in config are useful

    def _check_args(self):
        self.config.reproducibility.seed = get_seed(self.config.reproducibility.seed)
        if len(self.config.data.label_code) == 0:
            self.config.data.label_code = self.maps_manager.label_code

    def train(
        self,
        split_list: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Performs the training task for a defined list of splits.

        Parameters
        ----------
        split_list : Optional[List[int]] (optional, default=None)
            List of splits on which the training task is performed.
            Default trains all splits of the cross-validation.
        overwrite : bool (optional, default=False)
            If True, previously trained splits that are going to be trained
            are erased.

        Raises
        ------
        MAPSError
            If splits specified in input already exist and overwrite is False.
        """
        existing_splits = []

        split_manager = self.maps_manager._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            split_path = (
                self.maps_manager.maps_path / f"{self.maps_manager.split_name}-{split}"
            )
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

        if self.config.model.multi_network:
            self._train_multi(split_list, resume=False)
        elif self.config.ssda.ssda_network:
            self._train_ssda(split_list, resume=False)
        else:
            self._train_single(split_list, resume=False)

    def resume(
        self,
        split_list: Optional[List[int]] = None,
    ) -> None:
        """
        Resumes the training task for a defined list of splits.

        Parameters
        ----------
        split_list : Optional[List[int]] (optional, default=None)
            List of splits on which the training task is performed.
            If None, the training task is performed on all splits.

        Raises
        ------
        MAPSError
            If splits specified in input do not exist.
        """
        missing_splits = []
        split_manager = self.maps_manager._init_split_manager(split_list)

        for split in split_manager.split_iterator():
            if not (
                self.maps_manager.maps_path
                / f"{self.maps_manager.split_name}-{split}"
                / "tmp"
            ).is_dir():
                missing_splits.append(split)

        if len(missing_splits) > 0:
            raise MAPSError(
                f"Splits {missing_splits} were not initialized. "
                f"Please try train command on these splits and resume only others."
            )

        if self.config.model.multi_network:
            self._train_multi(split_list, resume=True)
        elif self.config.ssda.ssda_network:
            self._train_ssda(split_list, resume=True)
        else:
            self._train_single(split_list, resume=True)

    def _train_single(
        self,
        split_list: Optional[List[int]] = None,
        resume: bool = False,
    ) -> None:
        """
        Trains a single CNN for all inputs.

        Parameters
        ----------
        split_list : Optional[List[int]] (optional, default=None)
            List of splits on which the training task is performed.
            If None, performs training on all splits of the cross-validation.
        resume : bool (optional, default=False)
            If True, the job is resumed from checkpoint.
        """
        train_transforms, all_transforms = get_transforms(
            normalize=self.config.transforms.normalize,
            data_augmentation=self.config.transforms.data_augmentation,
            size_reduction=self.config.transforms.size_reduction,
            size_reduction_factor=self.config.transforms.size_reduction_factor,
        )
        split_manager = self.maps_manager._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(
                self.config.reproducibility.seed,
                self.config.reproducibility.deterministic,
                self.config.reproducibility.compensation,
            )

            split_df_dict = split_manager[split]

            logger.debug("Loading training data...")
            data_train = return_dataset(
                self.config.data.caps_directory,
                split_df_dict["train"],
                self.config.data.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.config.data.multi_cohort,
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )
            logger.debug("Loading validation data...")
            data_valid = return_dataset(
                self.config.data.caps_directory,
                split_df_dict["validation"],
                self.config.data.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.config.data.multi_cohort,
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )
            train_sampler = self.maps_manager.task_manager.generate_sampler(
                data_train,
                self.config.dataloader.sampler,
                dp_degree=cluster.world_size,
                rank=cluster.rank,
            )
            logger.debug(
                f"Getting train and validation loader with batch size {self.config.dataloader.batch_size}"
            )
            train_loader = DataLoader(
                data_train,
                batch_size=self.config.dataloader.batch_size,
                sampler=train_sampler,
                num_workers=self.config.dataloader.n_proc,
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
                batch_size=self.config.dataloader.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader.n_proc,
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
                self.maps_manager._ensemble_prediction(
                    "train",
                    split,
                    self.config.validation.selection_metrics,
                )
                self.maps_manager._ensemble_prediction(
                    "validation",
                    split,
                    self.config.validation.selection_metrics,
                )

                self._erase_tmp(split)

    def _train_multi(
        self,
        split_list: Optional[List[int]] = None,
        resume: bool = False,
    ) -> None:
        """
        Trains a CNN per element in the image (e.g. per slice).

        Parameters
        ----------
        split_list : Optional[List[int]] (optional, default=None)
            List of splits on which the training task is performed.
            If None, performs training on all splits of the cross-validation.
        resume : bool (optional, default=False)
            If True, the job is resumed from checkpoint.
        """
        train_transforms, all_transforms = get_transforms(
            normalize=self.config.transforms.normalize,
            data_augmentation=self.config.transforms.data_augmentation,
            size_reduction=self.config.transforms.size_reduction,
            size_reduction_factor=self.config.transforms.size_reduction_factor,
        )

        split_manager = self.maps_manager._init_split_manager(split_list)
        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(
                self.config.reproducibility.seed,
                self.config.reproducibility.deterministic,
                self.config.reproducibility.compensation,
            )

            split_df_dict = split_manager[split]

            first_network = 0
            if resume:
                training_logs = [
                    int(network_folder.split("-")[1])
                    for network_folder in list(
                        (
                            self.maps_manager.maps_path
                            / f"{self.maps_manager.split_name}-{split}"
                            / "training_logs"
                        ).iterdir()
                    )
                ]
                first_network = max(training_logs)
                if not (self.maps_manager.maps_path / "tmp").is_dir():
                    first_network += 1
                    resume = False

            for network in range(first_network, self.maps_manager.num_networks):
                logger.info(f"Train network {network}")

                data_train = return_dataset(
                    self.config.data.caps_directory,
                    split_df_dict["train"],
                    self.config.data.preprocessing_dict,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    multi_cohort=self.config.data.multi_cohort,
                    label=self.config.data.label,
                    label_code=self.config.data.label_code,
                    cnn_index=network,
                )
                data_valid = return_dataset(
                    self.config.data.caps_directory,
                    split_df_dict["validation"],
                    self.config.data.preprocessing_dict,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    multi_cohort=self.config.data.multi_cohort,
                    label=self.config.data.label,
                    label_code=self.config.data.label_code,
                    cnn_index=network,
                )

                train_sampler = self.maps_manager.task_manager.generate_sampler(
                    data_train,
                    self.config.dataloader.sampler,
                    dp_degree=cluster.world_size,
                    rank=cluster.rank,
                )
                train_loader = DataLoader(
                    data_train,
                    batch_size=self.config.dataloader.batch_size,
                    sampler=train_sampler,
                    num_workers=self.config.dataloader.n_proc,
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
                    batch_size=self.config.dataloader.batch_size,
                    shuffle=False,
                    num_workers=self.config.dataloader.n_proc,
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
                self.maps_manager._ensemble_prediction(
                    "train",
                    split,
                    self.config.validation.selection_metrics,
                )
                self.maps_manager._ensemble_prediction(
                    "validation",
                    split,
                    self.config.validation.selection_metrics,
                )

                self._erase_tmp(split)

    def _train_ssda(
        self,
        split_list: Optional[List[int]] = None,
        resume: bool = False,
    ) -> None:
        """
        Trains a single CNN for a source and target domain using semi-supervised domain adaptation.

        Parameters
        ----------
        split_list : Optional[List[int]] (optional, default=None)
            List of splits on which the training task is performed.
            If None, performs training on all splits of the cross-validation.
        resume : bool (optional, default=False)
            If True, the job is resumed from checkpoint.
        """
        train_transforms, all_transforms = get_transforms(
            normalize=self.config.transforms.normalize,
            data_augmentation=self.config.transforms.data_augmentation,
            size_reduction=self.config.transforms.size_reduction,
            size_reduction_factor=self.config.transforms.size_reduction_factor,
        )

        split_manager = self.maps_manager._init_split_manager(split_list)
        split_manager_target_lab = self.maps_manager._init_split_manager(
            split_list, True
        )

        for split in split_manager.split_iterator():
            logger.info(f"Training split {split}")
            seed_everything(
                self.config.reproducibility.seed,
                self.config.reproducibility.deterministic,
                self.config.reproducibility.compensation,
            )

            split_df_dict = split_manager[split]
            split_df_dict_target_lab = split_manager_target_lab[split]

            logger.debug("Loading source training data...")
            data_train_source = return_dataset(
                self.config.data.caps_directory,
                split_df_dict["train"],
                self.config.data.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.config.data.multi_cohort,
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )

            logger.debug("Loading target labelled training data...")
            data_train_target_labeled = return_dataset(
                Path(self.config.ssda.caps_target),  # TO CHECK
                split_df_dict_target_lab["train"],
                self.config.ssda.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,  # A checker
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )
            from torch.utils.data import ConcatDataset

            combined_dataset = ConcatDataset(
                [data_train_source, data_train_target_labeled]
            )

            logger.debug("Loading target unlabelled training data...")
            data_target_unlabeled = return_dataset(
                Path(self.config.ssda.caps_target),
                pd.read_csv(self.config.ssda.tsv_target_unlab, sep="\t"),
                self.config.ssda.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,  # A checker
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )

            logger.debug("Loading validation source data...")
            data_valid_source = return_dataset(
                self.config.data.caps_directory,
                split_df_dict["validation"],
                self.config.data.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.config.data.multi_cohort,
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )
            logger.debug("Loading validation target labelled data...")
            data_valid_target_labeled = return_dataset(
                Path(self.config.ssda.caps_target),
                split_df_dict_target_lab["validation"],
                self.config.ssda.preprocessing_dict_target,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=False,
                label=self.config.data.label,
                label_code=self.config.data.label_code,
            )
            train_source_sampler = self.maps_manager.task_manager.generate_sampler(
                data_train_source, self.config.dataloader.sampler
            )

            logger.info(
                f"Getting train and validation loader with batch size {self.config.dataloader.batch_size}"
            )

            ## Oversampling of the target dataset
            from torch.utils.data import SubsetRandomSampler

            # Create index lists for target labeled dataset
            labeled_indices = list(range(len(data_train_target_labeled)))

            # Oversample the indices for the target labelled dataset to match the size of the labeled source dataset
            data_train_source_size = (
                len(data_train_source) // self.config.dataloader.batch_size
            )
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
                batch_size=self.config.dataloader.batch_size,
                sampler=train_source_sampler,
                # shuffle=True,  # len(data_train_source) < len(data_train_target_labeled),
                num_workers=self.config.dataloader.n_proc,
                worker_init_fn=pl_worker_init_function,
                drop_last=True,
            )
            logger.info(
                f"Train source loader size is {len(train_source_loader)*self.config.dataloader.batch_size}"
            )
            train_target_loader = DataLoader(
                data_train_target_labeled,
                batch_size=1,  # To limit the need of oversampling
                # sampler=train_target_sampler,
                sampler=labeled_sampler,
                num_workers=self.config.dataloader.n_proc,
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
                batch_size=self.config.dataloader.batch_size,
                num_workers=self.config.dataloader.n_proc,
                # sampler=unlabeled_sampler,
                worker_init_fn=pl_worker_init_function,
                shuffle=True,
                drop_last=True,
            )

            logger.info(
                f"Train target unlabeled loader size is {len(train_target_unl_loader)*self.config.dataloader.batch_size}"
            )

            valid_loader_source = DataLoader(
                data_valid_source,
                batch_size=self.config.dataloader.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader.n_proc,
            )
            logger.info(
                f"Validation loader source size is {len(valid_loader_source)*self.config.dataloader.batch_size}"
            )

            valid_loader_target = DataLoader(
                data_valid_target_labeled,
                batch_size=self.config.dataloader.batch_size,  # To check
                shuffle=False,
                num_workers=self.config.dataloader.n_proc,
            )
            logger.info(
                f"Validation loader target size is {len(valid_loader_target)*self.config.dataloader.batch_size}"
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

            self.maps_manager._ensemble_prediction(
                "train",
                split,
                self.config.validation.selection_metrics,
            )
            self.maps_manager._ensemble_prediction(
                "validation",
                split,
                self.config.validation.selection_metrics,
            )

            self._erase_tmp(split)

    def _train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        split: int,
        network: int = None,
        resume: bool = False,
        callbacks: List[Callback] = [],
    ):
        """
        Core function shared by train and resume.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader wrapping the training set.
        valid_loader : torch.utils.data.DataLoader
            DataLoader wrapping the validation set.
        split : int
            Index of the split trained.
        network : int (optional, default=None)
            Index of the network trained (used in multi-network setting only).
        resume : bool (optional, default=False)
            If True the job is resumed from the checkpoint.
        callbacks : List[Callback] (optional, default=[])
            List of callbacks to call during training.

        Raises
        ------
        Exception
            _description_
        """
        self._init_callbacks()
        model, beginning_epoch = self.maps_manager._init_model(
            split=split,
            resume=resume,
            transfer_path=self.config.transfer_learning.transfer_path,
            transfer_selection=self.config.transfer_learning.transfer_selection_metric,
            nb_unfrozen_layer=self.config.transfer_learning.nb_unfrozen_layer,
        )
        model = DDP(
            model,
            fsdp=self.config.computational.fully_sharded_data_parallel,
            amp=self.config.computational.amp,
        )
        criterion = self.maps_manager.task_manager.get_criterion(self.config.model.loss)

        optimizer = self._init_optimizer(model, split=split, resume=resume)
        self.callback_handler.on_train_begin(
            self.maps_manager.parameters,
            criterion=criterion,
            optimizer=optimizer,
            split=split,
            maps_path=self.maps_manager.maps_path,
        )

        model.train()
        train_loader.dataset.train()

        early_stopping = EarlyStopping(
            "min",
            min_delta=self.config.early_stopping.tolerance,
            patience=self.config.early_stopping.patience,
        )
        metrics_valid = {"loss": None}

        if cluster.master:
            log_writer = LogWriter(
                self.maps_manager.maps_path,
                self.maps_manager.task_manager.evaluation_metrics + ["loss"],
                split,
                resume=resume,
                beginning_epoch=beginning_epoch,
                network=network,
            )
            retain_best = RetainBest(
                selection_metrics=list(self.config.validation.selection_metrics)
            )
        epoch = beginning_epoch

        retain_best = RetainBest(
            selection_metrics=list(self.config.validation.selection_metrics)
        )

        scaler = GradScaler(enabled=self.maps_manager.std_amp)
        profiler = self._init_profiler()

        if self.config.callbacks.track_exp == "wandb":
            from clinicadl.utils.tracking_exp import WandB_handler

        if self.config.lr_scheduler.adaptive_learning_rate:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            # Initialize the ReduceLROnPlateau scheduler
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, verbose=True
            )

        scaler = GradScaler(enabled=self.config.computational.amp)
        profiler = self._init_profiler()

        while epoch < self.config.optimization.epochs and not early_stopping.step(
            metrics_valid["loss"]
        ):
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
                    update: bool = (
                        i + 1
                    ) % self.config.optimization.accumulation_steps == 0
                    sync = nullcontext() if update else model.no_sync()
                    with sync:
                        with autocast(enabled=self.maps_manager.std_amp):
                            _, loss_dict = model(data, criterion)
                        logger.debug(f"Train loss dictionary {loss_dict}")
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
                            self.config.validation.evaluation_steps != 0
                            and (i + 1) % self.config.validation.evaluation_steps == 0
                        ):
                            evaluation_flag = False

                            _, metrics_train = self.maps_manager.task_manager.test(
                                model,
                                train_loader,
                                criterion,
                                amp=self.maps_manager.std_amp,
                            )
                            _, metrics_valid = self.maps_manager.task_manager.test(
                                model,
                                valid_loader,
                                criterion,
                                amp=self.maps_manager.std_amp,
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
                                f"{self.config.data.mode} level training loss is {metrics_train['loss']} "
                                f"at the end of iteration {i}"
                            )
                            logger.info(
                                f"{self.config.data.mode} level validation loss is {metrics_valid['loss']} "
                                f"at the end of iteration {i}"
                            )

                    profiler.step()

                # If no step has been performed, raise Exception
                if step_flag:
                    raise Exception(
                        "The model has not been updated once in the epoch. The accumulation step may be too large."
                    )

                # If no evaluation has been performed, warn the user
                elif evaluation_flag and self.config.validation.evaluation_steps != 0:
                    logger.warning(
                        f"Your evaluation steps {self.config.validation.evaluation_steps} are too big "
                        f"compared to the size of the dataset. "
                        f"The model is evaluated only once at the end epochs."
                    )

                # Update weights one last time if gradients were computed without update
                if (i + 1) % self.config.optimization.accumulation_steps != 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # Always test the results and save them once at the end of the epoch
                model.zero_grad(set_to_none=True)
                logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

                _, metrics_train = self.maps_manager.task_manager.test(
                    model, train_loader, criterion, amp=self.maps_manager.std_amp
                )
                _, metrics_valid = self.maps_manager.task_manager.test(
                    model, valid_loader, criterion, amp=self.maps_manager.std_amp
                )

                model.train()
                train_loader.dataset.train()

            self.callback_handler.on_epoch_end(
                self.maps_manager.parameters,
                metrics_train=metrics_train,
                metrics_valid=metrics_valid,
                mode=self.config.data.mode,
                i=i,
            )

            model_weights = {
                "model": model.state_dict(),
                "epoch": epoch,
                "name": self.config.model.architecture,
            }
            optimizer_weights = {
                "optimizer": model.optim_state_dict(optimizer),
                "epoch": epoch,
                "name": self.config.model.architecture,
            }

            if cluster.master:
                # Save checkpoints and best models
                best_dict = retain_best.step(metrics_valid)
                self._write_weights(
                    model_weights,
                    best_dict,
                    split,
                    network=network,
                    save_all_models=self.config.reproducibility.save_all_models,
                )
                self._write_weights(
                    optimizer_weights,
                    None,
                    split,
                    filename="optimizer.pth.tar",
                    save_all_models=self.config.reproducibility.save_all_models,
                )
            dist.barrier()

            if self.config.lr_scheduler.adaptive_learning_rate:
                scheduler.step(
                    metrics_valid["loss"]
                )  # Update learning rate based on validation loss

            epoch += 1

        del model
        self.maps_manager._test_loader(
            train_loader,
            criterion,
            "train",
            split,
            self.config.validation.selection_metrics,
            amp=self.maps_manager.std_amp,
            network=network,
        )
        self.maps_manager._test_loader(
            valid_loader,
            criterion,
            "validation",
            split,
            self.config.validation.selection_metrics,
            amp=self.maps_manager.std_amp,
            network=network,
        )

        if self.maps_manager.task_manager.save_outputs:
            self.maps_manager._compute_output_tensors(
                train_loader.dataset,
                "train",
                split,
                self.config.validation.selection_metrics,
                nb_images=1,
                network=network,
            )
            self.maps_manager._compute_output_tensors(
                valid_loader.dataset,
                "validation",
                split,
                self.config.validation.selection_metrics,
                nb_images=1,
                network=network,
            )

        self.callback_handler.on_train_end(parameters=self.maps_manager.parameters)

    def _train_ssdann(
        self,
        train_source_loader: DataLoader,
        train_target_loader: DataLoader,
        train_target_unl_loader: DataLoader,
        valid_loader: DataLoader,
        valid_source_loader: DataLoader,
        split: int,
        network: Optional[Any] = None,
        resume: bool = False,
        evaluate_source: bool = True,  # TO MODIFY
    ):
        """
        _summary_

        Parameters
        ----------
        train_source_loader : torch.utils.data.DataLoader
            _description_
        train_target_loader : torch.utils.data.DataLoader
            _description_
        train_target_unl_loader : torch.utils.data.DataLoader
            _description_
        valid_loader : torch.utils.data.DataLoader
            _description_
        valid_source_loader : torch.utils.data.DataLoader
            _description_
        split : int
            _description_
        network : Optional[Any] (optional, default=None)
            _description_
        resume : bool (optional, default=False)
            _description_
        evaluate_source : bool (optional, default=True)
            _description_

        Raises
        ------
        Exception
            _description_
        """
        model, beginning_epoch = self.maps_manager._init_model(
            split=split,
            resume=resume,
            transfer_path=self.config.transfer_learning.transfer_path,
            transfer_selection=self.config.transfer_learning.transfer_selection_metric,
        )

        criterion = self.maps_manager.task_manager.get_criterion(self.config.model.loss)
        logger.debug(f"Criterion for {self.config.network_task} is {criterion}")
        optimizer = self._init_optimizer(model, split=split, resume=resume)

        logger.debug(f"Optimizer used for training is optimizer")

        model.train()
        train_source_loader.dataset.train()
        train_target_loader.dataset.train()
        train_target_unl_loader.dataset.train()

        early_stopping = EarlyStopping(
            "min",
            min_delta=self.config.early_stopping.tolerance,
            patience=self.config.early_stopping.patience,
        )

        metrics_valid_target = {"loss": None}
        metrics_valid_source = {"loss": None}

        log_writer = LogWriter(
            self.maps_manager.maps_path,
            self.maps_manager.task_manager.evaluation_metrics + ["loss"],
            split,
            resume=resume,
            beginning_epoch=beginning_epoch,
            network=network,
        )
        epoch = log_writer.beginning_epoch

        retain_best = RetainBest(
            selection_metrics=list(self.config.validation.selection_metrics)
        )
        import numpy as np

        while epoch < self.config.optimization.epochs and not early_stopping.step(
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
                logger.debug(f"Train loss dictionary {loss_dict}")
                loss = loss_dict["loss"]
                loss.backward()
                if (i + 1) % self.config.optimization.accumulation_steps == 0:
                    step_flag = False
                    optimizer.step()
                    optimizer.zero_grad()

                    del loss

                    # Evaluate the model only when no gradients are accumulated
                    if (
                        self.config.validation.evaluation_steps != 0
                        and (i + 1) % self.config.validation.evaluation_steps == 0
                    ):
                        evaluation_flag = False

                        # Evaluate on target data
                        logger.info("Evaluation on target data")
                        (
                            _,
                            metrics_train_target,
                        ) = self.maps_manager.task_manager.test_da(
                            model,
                            train_target_loader,
                            criterion,
                            alpha,
                            target=True,
                        )  # TO CHECK

                        (
                            _,
                            metrics_valid_target,
                        ) = self.maps_manager.task_manager.test_da(
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
                            f"{self.config.data.mode} level training loss for target data is {metrics_train_target['loss']} "
                            f"at the end of iteration {i}"
                        )
                        logger.info(
                            f"{self.config.data.mode} level validation loss for target data is {metrics_valid_target['loss']} "
                            f"at the end of iteration {i}"
                        )

                        # Evaluate on source data
                        logger.info("Evaluation on source data")
                        (
                            _,
                            metrics_train_source,
                        ) = self.maps_manager.task_manager.test_da(
                            model, train_source_loader, criterion, alpha
                        )
                        (
                            _,
                            metrics_valid_source,
                        ) = self.maps_manager.task_manager.test_da(
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
                            f"{self.config.data.mode} level training loss for source data is {metrics_train_source['loss']} "
                            f"at the end of iteration {i}"
                        )
                        logger.info(
                            f"{self.config.data.mode} level validation loss for source data is {metrics_valid_source['loss']} "
                            f"at the end of iteration {i}"
                        )

            # If no step has been performed, raise Exception
            if step_flag:
                raise Exception(
                    "The model has not been updated once in the epoch. The accumulation step may be too large."
                )

            # If no evaluation has been performed, warn the user
            elif evaluation_flag and self.config.validation.evaluation_steps != 0:
                logger.warning(
                    f"Your evaluation steps {self.config.validation.evaluation_steps} are too big "
                    f"compared to the size of the dataset. "
                    f"The model is evaluated only once at the end epochs."
                )

            # Update weights one last time if gradients were computed without update
            if (i + 1) % self.config.optimization.accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            # Always test the results and save them once at the end of the epoch
            model.zero_grad()
            logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

            if evaluate_source:
                logger.info(
                    f"Evaluate source data at the end of the epoch {epoch} with alpha: {alpha}."
                )
                _, metrics_train_source = self.maps_manager.task_manager.test_da(
                    model,
                    train_source_loader,
                    criterion,
                    alpha,
                    True,
                    False,
                )
                _, metrics_valid_source = self.maps_manager.task_manager.test_da(
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
                    f"{self.config.data.mode} level training loss for source data is {metrics_train_source['loss']} "
                    f"at the end of iteration {i}"
                )
                logger.info(
                    f"{self.config.data.mode} level validation loss for source data is {metrics_valid_source['loss']} "
                    f"at the end of iteration {i}"
                )

            _, metrics_train_target = self.maps_manager.task_manager.test_da(
                model,
                train_target_loader,
                criterion,
                alpha,
                target=True,
            )
            _, metrics_valid_target = self.maps_manager.task_manager.test_da(
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
                f"{self.config.data.mode} level training loss for target data is {metrics_train_target['loss']} "
                f"at the end of iteration {i}"
            )
            logger.info(
                f"{self.config.data.mode} level validation loss for target data is {metrics_valid_target['loss']} "
                f"at the end of iteration {i}"
            )

            # Save checkpoints and best models
            best_dict = retain_best.step(metrics_valid_target)
            self._write_weights(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "name": self.config.model.architecture,
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
                    "name": self.config.optimizer,
                },
                None,
                split,
                filename="optimizer.pth.tar",
                save_all_models=False,
            )

            epoch += 1

        self.maps_manager._test_loader_ssda(
            train_target_loader,
            criterion,
            data_group="train",
            split=split,
            selection_metrics=self.config.validation.selection_metrics,
            network=network,
            target=True,
            alpha=0,
        )
        self.maps_manager._test_loader_ssda(
            valid_loader,
            criterion,
            data_group="validation",
            split=split,
            selection_metrics=self.config.validation.selection_metrics,
            network=network,
            target=True,
            alpha=0,
        )

        if self.maps_manager.task_manager.save_outputs:
            self.maps_manager._compute_output_tensors(
                train_target_loader.dataset,
                "train",
                split,
                self.config.validation.selection_metrics,
                nb_images=1,
                network=network,
            )
            self.maps_manager._compute_output_tensors(
                train_target_loader.dataset,
                "validation",
                split,
                self.config.validation.selection_metrics,
                nb_images=1,
                network=network,
            )

    def _init_callbacks(self) -> None:
        """
        Initializes training callbacks.
        """
        from clinicadl.utils.callbacks.callbacks import CallbacksHandler, LoggerCallback

        # if self.callbacks is None:
        #     self.callbacks = [Callback()]

        self.callback_handler = CallbacksHandler()  # callbacks=self.callbacks)

        if self.config.callbacks.emissions_calculator:
            from clinicadl.utils.callbacks.callbacks import CodeCarbonTracker

            self.callback_handler.add_callback(CodeCarbonTracker())

        if self.config.callbacks.track_exp:
            from clinicadl.utils.callbacks.callbacks import Tracker

            self.callback_handler.add_callback(Tracker)

        self.callback_handler.add_callback(LoggerCallback())
        # self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def _init_optimizer(
        self,
        model: DDP,
        split: int = None,
        resume: bool = False,
    ) -> torch.optim.Optimizer:
        """
        Initializes the optimizer.

        Parameters
        ----------
        model : clinicadl.utils.maps_manager.ddp.DDP
            The parallelizer.
        split : int (optional, default=None)
            The split considered. Should not be None if resume is True, but is
            useless when resume is False.
        resume : bool (optional, default=False)
            If True, uses checkpoint to recover optimizer's old state.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer.
        """

        optimizer_cls = getattr(torch.optim, self.config.optimizer.optimizer)
        parameters = filter(lambda x: x.requires_grad, model.parameters())
        optimizer_kwargs = dict(
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )

        optimizer = optimizer_cls(parameters, **optimizer_kwargs)

        if resume:
            checkpoint_path = (
                self.maps_manager.maps_path
                / f"{self.maps_manager.split_name}-{split}"
                / "tmp"
                / "optimizer.pth.tar"
            )
            checkpoint_state = torch.load(checkpoint_path, map_location=model.device)
            model.load_optim_state_dict(optimizer, checkpoint_state["optimizer"])

        return optimizer

    def _init_profiler(self) -> torch.profiler.profile:
        """
        Initializes the profiler.

        Returns
        -------
        torch.profiler.profile
            Profiler context manager.
        """
        if self.config.optimization.profiler:
            from clinicadl.utils.maps_manager.cluster.profiler import (
                ProfilerActivity,
                profile,
                schedule,
                tensorboard_trace_handler,
            )

            time = datetime.now().strftime("%H:%M:%S")
            filename = [self.maps_manager.maps_path / "profiler" / f"clinicadl_{time}"]
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

    def _erase_tmp(self, split: int):
        """
        Erases checkpoints of the model and optimizer at the end of training.

        Parameters
        ----------
        split : int
            The split on which the model has been trained.
        """
        tmp_path = (
            self.maps_manager.maps_path
            / f"{self.maps_manager.split_name}-{split}"
            / "tmp"
        )
        shutil.rmtree(tmp_path)

    def _write_weights(
        self,
        state: Dict[str, Any],
        metrics_dict: Optional[Dict[str, bool]],
        split: int,
        network: int = None,
        filename: str = "checkpoint.pth.tar",
        save_all_models: bool = False,
    ) -> None:
        """
        Update checkpoint and save the best model according to a set of
        metrics.

        Parameters
        ----------
        state : Dict[str, Any]
            The state of the training (model weights, epoch, etc.).
        metrics_dict : Optional[Dict[str, bool]]
            The output of RetainBest step. If None, only the checkpoint
            is saved.
        split : int
            The split number.
        network : int (optional, default=None)
            The network number (multi-network framework).
        filename : str (optional, default="checkpoint.pth.tar")
            The name of the checkpoint file.
        save_all_models : bool (optional, default=False)
            Whether to save model weights at every epoch.
            If False, only the best model will be saved.
        """
        checkpoint_dir = (
            self.maps_manager.maps_path
            / f"{self.maps_manager.split_name}-{split}"
            / "tmp"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / filename
        torch.save(state, checkpoint_path)

        if save_all_models:
            all_models_dir = (
                self.maps_manager.maps_path
                / f"{self.maps_manager.split_name}-{split}"
                / "all_models"
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
                    self.maps_manager.maps_path
                    / f"{self.maps_manager.split_name}-{split}"
                    / f"best-{metric_name}"
                )
                if metric_bool:
                    metric_path.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(checkpoint_path, metric_path / best_filename)
