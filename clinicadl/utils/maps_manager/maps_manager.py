import json
import shutil
import subprocess
from datetime import datetime
from logging import getLogger
from os import listdir, makedirs, path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from clinicadl.extract.extract_utils import compute_folder_and_file_type
from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    load_data_test,
    return_dataset,
)
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.maps_manager.logwriter import LogWriter, setup_logging
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.network.network import Network
from clinicadl.utils.seed import get_seed, pl_worker_init_function, seed_everything

logger = getLogger("clinicadl")


level_list: List[str] = ["warning", "info", "debug"]
# TODO: replace "fold" with "split"
# TODO save weights on CPU for better compatibility


class MapsManager:
    def __init__(
        self,
        maps_path: str,
        parameters: Dict[str, Any] = None,
        verbose: str = "warning",
    ):
        """
        Args:
            maps_path: path of the MAPS
            parameters: parameters of the training step. If given a new MAPS is created.
            verbose: Logging level ("debug", "info", "warning")
        """
        self.maps_path = maps_path
        if verbose not in level_list:
            raise ValueError(f"verbose value {verbose} must be in {level_list}.")
        setup_logging(level_list.index(verbose))

        # Existing MAPS
        if parameters is None:
            if not path.exists(path.join(maps_path, "maps.json")):
                raise ValueError(
                    f"MAPS was not found at {maps_path}."
                    f"To initiate a new MAPS please give a train_dict."
                )
            self.parameters = self.get_parameters()
            self.task_manager = self._init_task_manager(n_classes=self.output_size)

        # Initiate MAPS
        else:
            if (
                path.exists(maps_path) and not path.isdir(maps_path)  # Non-folder file
            ) or (
                path.isdir(maps_path) and listdir(maps_path)  # Non empty folder
            ):
                raise ValueError(
                    f"You are trying to create a new MAPS at {maps_path} but "
                    f"this already corresponds to a file or a non-empty folder. \n"
                    f"Please remove it or choose another location."
                )
            makedirs(path.join(self.maps_path, "groups"))
            logger.info(f"A new MAPS was created at {maps_path}")
            self._check_args(parameters)
            self.write_parameters(self.maps_path, self.parameters)
            self._write_requirements_version()
            self._write_training_data()
            self._write_train_val_groups()

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"'MapsManager' object has no attribute '{name}'")

    def train(self, folds=None, overwrite=False):
        """
        Performs the training task for a defined list of folds

        Args:
            folds (list[int]): list of folds on which the training task is performed.
                Default trains all folds.
            overwrite (bool): If True previously trained folds that are going to be trained
                are erased.

        Raises:
            ValueError: If folds specified in input already exist and overwrite is False.
        """
        existing_folds = []

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            fold_path = path.join(self.maps_path, f"fold-{fold}")
            if path.exists(fold_path):
                if overwrite:
                    shutil.rmtree(fold_path)
                else:
                    existing_folds.append(fold)

        if len(existing_folds) > 0:
            raise ValueError(
                f"Folds {existing_folds} already exist. Please "
                f"specify a list of folds not intersecting the previous list, "
                f"or use overwrite to erase previously trained folds."
            )
        if self.multi_network:
            self._train_multi(folds, resume=False)
        else:
            self._train_single(folds, resume=False)

    def resume(self, folds=None):
        """
        Resumes the training task for a defined list of folds

        Args:
            folds (list[int]): list of folds on which the training task is performed.
                Default trains all folds.

        Raises:
            ValueError: If folds specified in input do not exist.
        """
        missing_folds = []
        split_manager = self._init_split_manager(folds)

        for fold in split_manager.fold_iterator():
            if not path.exists(path.join(self.maps_path, f"fold-{fold}", "tmp")):
                missing_folds.append(fold)

        if len(missing_folds) > 0:
            raise ValueError(
                f"Folds {missing_folds} were not initialized. "
                f"Please try train command on these folds and resume only others."
            )

        if self.multi_network:
            self._train_multi(folds, resume=True)
        else:
            self._train_single(folds, resume=True)

    def predict(
        self,
        data_group,
        caps_directory=None,
        tsv_path=None,
        folds=None,
        selection_metrics=None,
        multi_cohort=False,
        diagnoses=(),
        use_labels=True,
        batch_size=None,
        num_workers=None,
        use_cpu=None,
        overwrite=False,
    ):
        """
        Performs the prediction task on a subset of caps_directory defined in a TSV file.

        Args:
            data_group (str): name of the data group tested.
            caps_directory (str): path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default will load the value of an existing data group
            tsv_path (str): path to a TSV file containing the list of participants and sessions to test.
                Default will load the DataFrame of an existing data group
            folds (list[int]): list of folds to test. Default perform prediction on all folds available.
            selection_metrics (list[str]): list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
            multi_cohort (bool): If True considers that tsv_path is the path to a multi-cohort TSV.
            diagnoses (list[str]): List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            batch_size (int): If given, sets the value of batch_size, else use the same as in training step.
            num_workers (int): If given, sets the value of num_workers, else use the same as in training step.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            overwrite (bool): If True erase the occurrences of data_group.

        Raises:
            ValueError:
                when trying to overwrite train or validation data groups
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
        """
        if folds is None:
            folds = self._find_folds()

        _, all_transforms = get_transforms(
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.diagnoses,
                multi_cohort=multi_cohort,
            )

        criterion = self.task_manager.get_criterion()

        for fold in folds:
            group_df, group_parameters = self._check_data_group(
                data_group, fold, caps_directory, group_df, multi_cohort, overwrite
            )

            if self.multi_network:
                for network in range(self.num_networks):
                    data_test = return_dataset(
                        group_parameters["caps_directory"],
                        group_df,
                        self.preprocessing_dict,
                        all_transformations=all_transforms,
                        multi_cohort=group_parameters["multi_cohort"],
                        label_presence=use_labels,
                        label=self.label,
                        label_code=self.label_code,
                        cnn_index=network,
                    )
                    test_loader = DataLoader(
                        data_test,
                        batch_size=batch_size
                        if batch_size is not None
                        else self.batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        if num_workers is not None
                        else self.num_workers,
                    )

                    self._test_loader(
                        test_loader,
                        criterion,
                        data_group,
                        fold,
                        selection_metrics
                        if selection_metrics is not None
                        else self._find_selection_metrics(fold),
                        use_labels=use_labels,
                        use_cpu=use_cpu,
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
                    label=self.label,
                    label_code=self.label_code,
                )
                test_loader = DataLoader(
                    data_test,
                    batch_size=batch_size
                    if batch_size is not None
                    else self.batch_size,
                    shuffle=False,
                    num_workers=num_workers
                    if num_workers is not None
                    else self.num_workers,
                )

                self._test_loader(
                    test_loader,
                    criterion,
                    data_group,
                    fold,
                    selection_metrics
                    if selection_metrics is not None
                    else self._find_selection_metrics(fold),
                    use_labels=use_labels,
                    use_cpu=use_cpu,
                )
            self._ensemble_prediction(data_group, fold, selection_metrics, use_labels)

    def save_tensors(
        self,
        data_group,
        caps_directory=None,
        tsv_path=None,
        folds=None,
        selection_metrics=None,
        multi_cohort=False,
        diagnoses=(),
        use_cpu=None,
        overwrite=False,
    ):

        """
        Computes and saves the input and output tensors of the set data_group.

        Args:
            data_group (str): name of the data group on which extraction is performed.
            caps_directory (str): path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default will load the value of an existing data group.
            tsv_path (str): path to a TSV file containing the list of participants and sessions to test.
                Default will load the DataFrame of an existing data group.
            folds (list[int]): list of folds to test. Default perform prediction on all folds available.
            selection_metrics (list[str]): list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
            multi_cohort (bool): If True considers that tsv_path is the path to a multi-cohort TSV.
            diagnoses (list[str]): List of diagnoses to load if tsv_path is a split_directory.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            overwrite (bool): If True erase the occurrences of data_group.

        Raises:
            ValueError:
                when trying to overwrite train or validation data groups
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
        """
        if folds is None:
            folds = self._find_folds()

        _, all_transforms = get_transforms(
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.diagnoses,
                multi_cohort=multi_cohort,
            )

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            logger.info(f"Saving tensors of fold {fold}")

            group_df, group_parameters = self._check_data_group(
                data_group, fold, caps_directory, group_df, multi_cohort, overwrite
            )

            if selection_metrics is None:
                selection_metrics = self._find_selection_metrics(fold)

            if self.multi_network:
                for network in range(self.num_networks):
                    dataset = return_dataset(
                        group_parameters["caps_directory"],
                        group_df,
                        self.preprocessing_dict,
                        all_transformations=all_transforms,
                        multi_cohort=group_parameters["multi_cohort"],
                        label=self.label,
                        label_code=self.label_code,
                        cnn_index=network,
                    )
                    self._compute_output_tensors(
                        dataset,
                        data_group,
                        fold,
                        selection_metrics,
                        use_cpu=use_cpu,
                        network=network,
                    )

            else:
                dataset = return_dataset(
                    group_parameters["caps_directory"],
                    group_df,
                    self.preprocessing_dict,
                    all_transformations=all_transforms,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                )
                self._compute_output_tensors(
                    dataset,
                    data_group,
                    fold,
                    selection_metrics,
                    use_cpu=use_cpu,
                )

    def interpret(
        self,
        data_group,
        name,
        caps_directory=None,
        tsv_path=None,
        folds=None,
        selection_metrics=None,
        multi_cohort=False,
        diagnoses=(),
        target_node=0,
        save_individual=False,
        batch_size=None,
        num_workers=None,
        use_cpu=None,
        overwrite=False,
        overwrite_name=False,
    ):
        """
        Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.

        Args:
            data_group (str): name of the data group interpreted.
            name (str): name of the interpretation procedure.
            caps_directory (str): path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default will load the value of an existing data group.
            tsv_path (str): path to a TSV file containing the list of participants and sessions to test.
                Default will load the DataFrame of an existing data group.
            folds (list[int]): list of folds to interpret. Default perform interpretation on all folds available.
            selection_metrics (list[str]): list of selection metrics to interpret.
                Default performs the interpretation on all selection metrics available.
            multi_cohort (bool): If True considers that tsv_path is the path to a multi-cohort TSV.
            diagnoses (list[str]): List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            target_node (int): Node from which the interpretation is computed.
            save_individual (bool): If True saves the individual map of each participant / session couple.
            batch_size (int): If given, sets the value of batch_size, else use the same as in training step.
            num_workers (int): If given, sets the value of num_workers, else use the same as in training step.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            overwrite (bool): If True erase the occurrences of data_group.
            overwrite_name (bool): If True erase the occurrences of name.
        Raises:
            ValueError:
                when trying to overwrite train or validation data groups
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
                when name already exists and overwrite_name is False
        """

        from torch.utils.data import DataLoader

        from clinicadl.interpret.gradients import VanillaBackProp

        if folds is None:
            folds = self._find_folds()

        if self.multi_network:
            raise NotImplementedError(
                "The interpretation of multi-network framework is not implemented."
            )

        _, all_transforms = get_transforms(
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        group_df = None
        if tsv_path is not None:
            group_df = load_data_test(
                tsv_path,
                diagnoses if len(diagnoses) != 0 else self.diagnoses,
                multi_cohort=multi_cohort,
            )

        for fold in folds:
            logger.info(f"Interpretation of fold {fold}")
            df_group, parameters_group = self._check_data_group(
                data_group, fold, caps_directory, group_df, multi_cohort, overwrite
            )

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
                num_workers=num_workers
                if num_workers is not None
                else self.num_workers,
            )

            if selection_metrics is None:
                selection_metrics = self._find_selection_metrics(fold)

            for selection_metric in selection_metrics:
                logger.info(f"Interpretation of metric {selection_metric}")
                results_path = path.join(
                    self.maps_path,
                    f"fold-{fold}",
                    f"best-{selection_metric}",
                    data_group,
                    f"interpret-{name}",
                )

                if path.exists(results_path):
                    if overwrite_name:
                        shutil.rmtree(results_path)
                    else:
                        raise ValueError(
                            f"Interpretation name {name} is already written. "
                            f"Please choose another name or set overwrite_name to True."
                        )
                makedirs(results_path)

                model, _ = self._init_model(
                    transfer_path=self.maps_path,
                    fold=fold,
                    transfer_selection=selection_metric,
                    use_cpu=use_cpu,
                )

                interpreter = VanillaBackProp(model)

                cum_maps = [0] * data_test.elem_per_image
                for data in test_loader:
                    images = data["image"].to(model.device)

                    map_pt = interpreter.generate_gradients(images, target_node)
                    for i in range(len(data["participant_id"])):
                        mode_id = data[f"{self.mode}_id"][i]
                        cum_maps[mode_id] += map_pt[i]
                        if save_individual:
                            single_path = path.join(
                                results_path,
                                f"participant-{data['participant_id'][i]}_session-{data['session_id'][i]}_"
                                f"{self.mode}-{data[f'{self.mode}_id'][i]}_map.pt",
                            )
                            torch.save(map_pt, single_path)
                for i, mode_map in enumerate(cum_maps):
                    mode_map /= len(data_test)
                    torch.save(
                        mode_map,
                        path.join(results_path, f"mean_{self.mode}-{i}_map.pt"),
                    )

    ###################################
    # High-level functions templates  #
    ###################################
    def _train_single(self, folds=None, resume=False):
        """
        Trains a single CNN for all inputs.

        Args:
            folds (list[int]): list of folds that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        from torch.utils.data import DataLoader

        train_transforms, all_transforms = get_transforms(
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            logger.info(f"Training fold {fold}")
            seed_everything(self.seed, self.deterministic, self.compensation)

            fold_df_dict = split_manager[fold]

            data_train = return_dataset(
                self.caps_directory,
                fold_df_dict["train"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )
            data_valid = return_dataset(
                self.caps_directory,
                fold_df_dict["validation"],
                self.preprocessing_dict,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
            )

            train_sampler = self.task_manager.generate_sampler(data_train, self.sampler)

            train_loader = DataLoader(
                data_train,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
                worker_init_fn=pl_worker_init_function,
            )

            valid_loader = DataLoader(
                data_valid,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            self._train(
                train_loader,
                valid_loader,
                fold,
                resume=resume,
            )

            self._ensemble_prediction(
                "train",
                fold,
                self.selection_metrics,
            )
            self._ensemble_prediction(
                "validation",
                fold,
                self.selection_metrics,
            )

            self._erase_tmp(fold)

    def _train_multi(self, folds=None, resume=False):
        """
        Trains a single CNN per element in the image.

        Args:
            folds (list[int]): list of folds that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        from torch.utils.data import DataLoader

        train_transforms, all_transforms = get_transforms(
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            logger.info(f"Training fold {fold}")
            seed_everything(self.seed, self.deterministic, self.compensation)

            fold_df_dict = split_manager[fold]

            first_network = 0
            if resume:
                training_logs = [
                    int(network_folder.split("-")[1])
                    for network_folder in listdir(
                        path.join(self.maps_path, f"fold-{fold}", "training_logs")
                    )
                ]
                first_network = max(training_logs)
                if not path.exists(path.join(self.maps_path, "tmp")):
                    first_network += 1
                    resume = False

            for network in range(first_network, self.num_networks):
                logger.info(f"Train network {network}")

                data_train = return_dataset(
                    self.caps_directory,
                    fold_df_dict["train"],
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
                    fold_df_dict["validation"],
                    self.preprocessing_dict,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                    cnn_index=network,
                )

                train_sampler = self.task_manager.generate_sampler(
                    data_train, self.sampler
                )

                train_loader = DataLoader(
                    data_train,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                    num_workers=self.num_workers,
                    worker_init_fn=pl_worker_init_function,
                )

                valid_loader = DataLoader(
                    data_valid,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

                self._train(
                    train_loader,
                    valid_loader,
                    fold,
                    network,
                    resume=resume,
                )
                resume = False

            self._ensemble_prediction(
                "train",
                fold,
                self.selection_metrics,
            )
            self._ensemble_prediction(
                "validation",
                fold,
                self.selection_metrics,
            )

            self._erase_tmp(fold)

    def _train(
        self,
        train_loader,
        valid_loader,
        fold,
        network=None,
        resume=False,
    ):
        """
        Core function shared by train and resume.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader wrapping the training set.
            valid_loader (torch.utils.data.DataLoader): DataLoader wrapping the validation set.
            fold (int): Index of the fold trained.
            network (int): Index of the network trained (used in multi-network setting only).
            resume (bool): If True the job is resumed from the checkpoint.
        """

        model, beginning_epoch = self._init_model(
            fold=fold,
            resume=resume,
            transfer_path=self.transfer_path,
            transfer_selection=self.transfer_selection_metric,
        )
        criterion = self.task_manager.get_criterion()
        optimizer = self._init_optimizer(model, fold=fold, resume=resume)

        model.train()
        train_loader.dataset.train()

        early_stopping = EarlyStopping(
            "min", min_delta=self.tolerance, patience=self.patience
        )
        metrics_valid = {"loss": None}

        log_writer = LogWriter(
            self.maps_path,
            self.task_manager.evaluation_metrics + ["loss"],
            fold,
            resume=resume,
            beginning_epoch=beginning_epoch,
            network=network,
        )
        epoch = log_writer.beginning_epoch

        retain_best = RetainBest(selection_metrics=self.selection_metrics)

        while epoch < self.epochs and not early_stopping.step(metrics_valid["loss"]):
            logger.info(f"Beginning epoch {epoch}.")

            model.zero_grad()
            evaluation_flag, step_flag = True, True

            for i, data in enumerate(train_loader):

                _, loss = model.compute_outputs_and_loss(data, criterion)
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

                        _, metrics_train = self.task_manager.test(
                            model, train_loader, criterion
                        )
                        _, metrics_valid = self.task_manager.test(
                            model, valid_loader, criterion
                        )

                        model.train()
                        train_loader.dataset.train()

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

            _, metrics_train = self.task_manager.test(model, train_loader, criterion)
            _, metrics_valid = self.task_manager.test(model, valid_loader, criterion)

            model.train()
            train_loader.dataset.train()

            log_writer.step(epoch, i, metrics_train, metrics_valid, len(train_loader))
            logger.info(
                f"{self.mode} level training loss is {metrics_train['loss']} "
                f"at the end of iteration {i}"
            )
            logger.info(
                f"{self.mode} level validation loss is {metrics_valid['loss']} "
                f"at the end of iteration {i}"
            )

            # Save checkpoints and best models
            best_dict = retain_best.step(metrics_valid)
            self._write_weights(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "name": self.architecture,
                },
                best_dict,
                fold,
                network=network,
            )
            self._write_weights(
                {
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "name": self.optimizer,
                },
                None,
                fold,
                filename="optimizer.pth.tar",
            )

            epoch += 1

        self._test_loader(
            train_loader,
            criterion,
            "train",
            fold,
            self.selection_metrics,
            network=network,
        )
        self._test_loader(
            valid_loader,
            criterion,
            "validation",
            fold,
            self.selection_metrics,
            network=network,
        )

        if self.task_manager.save_outputs:
            self._compute_output_tensors(
                train_loader.dataset,
                "train",
                fold,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )
            self._compute_output_tensors(
                train_loader.dataset,
                "validation",
                fold,
                self.selection_metrics,
                nb_images=1,
                network=network,
            )

    def _test_loader(
        self,
        dataloader,
        criterion,
        data_group,
        fold,
        selection_metrics,
        use_labels=True,
        use_cpu=None,
        network=None,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            data_group (str): name of the data group used for the testing task.
            fold (int): Index of the fold used to train the model tested.
            selection_metrics (list[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:

            log_dir = path.join(
                self.maps_path, f"fold-{fold}", f"best-{selection_metric}", data_group
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
                fold=fold,
                transfer_selection=selection_metric,
                use_cpu=use_cpu,
                network=network,
            )

            prediction_df, metrics = self.task_manager.test(
                model, dataloader, criterion, use_labels=use_labels
            )
            if use_labels:
                if network is not None:
                    metrics[f"{self.mode}_id"] = network
                logger.info(
                    f"{self.mode} level {data_group} loss is {metrics['loss']} for model selected on {selection_metric}"
                )

            # Replace here
            self._mode_level_to_tsv(
                prediction_df, metrics, fold, selection_metric, data_group=data_group
            )

    def _compute_output_tensors(
        self,
        dataset,
        data_group,
        fold,
        selection_metrics,
        nb_images=None,
        use_cpu=None,
        network=None,
    ):
        """
        Compute the output tensors and saves them in the MAPS.

        Args:
            dataset (clinicadl.utils.caps_dataset.data.CapsDataset): wrapper of the data set.
            data_group (str): name of the data group used for the task.
            fold (int): Fold number.
            selection_metrics (list[str]): metrics used for model selection.
            nb_images (int): number of full images to write. Default computes the outputs of the whole data set.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:
            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.maps_path,
                fold=fold,
                transfer_selection=selection_metric,
                use_cpu=use_cpu,
                network=network,
            )

            tensor_path = path.join(
                self.maps_path,
                f"fold-{fold}",
                f"best-{selection_metric}",
                data_group,
                "tensors",
            )
            makedirs(tensor_path, exist_ok=True)

            if nb_images is None:  # Compute outputs for the whole data set
                nb_modes = len(dataset)
            else:
                nb_modes = nb_images * dataset.elem_per_image

            for i in range(nb_modes):
                data = dataset[i]
                image = data["image"]
                output = (
                    model.predict(image.unsqueeze(0).to(model.device)).squeeze(0).cpu()
                )
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{self.mode}_id"]
                input_filename = (
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_input.pt"
                )
                output_filename = (
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_output.pt"
                )
                torch.save(image, path.join(tensor_path, input_filename))
                torch.save(output, path.join(tensor_path, output_filename))

    def _ensemble_prediction(
        self,
        data_group,
        fold,
        selection_metrics,
        use_labels=True,
    ):
        """Computes the results on the image-level."""

        if selection_metrics is None:
            selection_metrics = self._find_selection_metrics(fold)

        for selection_metric in selection_metrics:
            # Soft voting
            if self.num_networks > 1:
                self._ensemble_to_tsv(
                    fold,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
            elif self.mode != "image":
                self._mode_to_image_tsv(
                    fold,
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
        TODO: create independent class for train_parameters check
        """
        mandatory_arguments = [
            "caps_directory",
            "tsv_path",
            "preprocessing_dict",
            "mode",
            "network_task",
        ]

        for arg in mandatory_arguments:
            if arg not in parameters:
                raise ValueError(
                    f"The values of mandatory arguments {mandatory_arguments} should be set. "
                    f"No value was given for {arg}."
                )

        self.parameters = parameters

        _, transformations = get_transforms(self.minmaxnormalization)

        split_manager = self._init_split_manager(None)
        train_df = split_manager[0]["train"]
        if "label" not in self.parameters:
            self.parameters["label"] = None

        self.task_manager = self._init_task_manager(df=train_df)

        if self.parameters["architecture"] == "default":
            self.parameters["architecture"] = self.task_manager.get_default_network()
        if "selection_threshold" not in self.parameters:
            self.parameters["selection_threshold"] = None
        label_code = self.task_manager.generate_label_code(train_df, self.label)
        full_dataset = return_dataset(
            self.caps_directory,
            train_df,
            self.preprocessing_dict,
            multi_cohort=self.multi_cohort,
            label=self.label,
            label_code=label_code,
            train_transformations=None,
            all_transformations=transformations,
        )
        self.parameters.update(
            {
                "num_networks": full_dataset.elem_per_image,
                "label_code": label_code,
                "output_size": self.task_manager.output_size(
                    full_dataset.size, full_dataset.df, self.label
                ),
                "input_size": full_dataset.size,
            }
        )

        self.parameters["seed"] = get_seed(self.parameters["seed"])

        if self.parameters["num_networks"] < 2 and self.multi_network:
            raise ValueError(
                f"Invalid training arguments: cannot train a multi-network "
                f"framework with only {self.parameters['num_networks']} element "
                f"per image."
            )
        possible_selection_metrics_set = set(self.task_manager.evaluation_metrics) | {
            "loss"
        }
        if not set(self.parameters["selection_metrics"]).issubset(
            possible_selection_metrics_set
        ):
            raise ValueError(
                f"Selection metrics {self.parameters['selection_metrics']} "
                f"must be a subset of metrics used for evaluation "
                f"{possible_selection_metrics_set}."
            )

        # TODO: add default values manager
        # click passing context @click.command / @click.passcontext (config.json)
        # or default parameters in click --> from config_param import learning_rate --> @learning_rate

    def _find_folds(self):
        """Find which folds were trained in the MAPS."""
        return [
            int(fold[5::])
            for fold in listdir(self.maps_path)
            if fold.startswith("fold-")
        ]

    def _find_selection_metrics(self, fold):
        """Find which selection metrics are available in MAPS for a given fold."""
        fold_path = path.join(self.maps_path, f"fold-{fold}")
        if not path.exists(fold_path):
            raise ValueError(
                f"Training of fold {fold} was not performed."
                f"Please execute maps_manager.train(folds={fold})"
            )

        return [metric[5::] for metric in listdir(fold_path) if metric[:5:] == "best-"]

    def _check_selection_metric(self, fold, selection_metric=None):
        """Check that a given selection metric is available for a given fold."""
        available_metrics = self._find_selection_metrics(fold)
        if selection_metric is None:
            if len(available_metrics) > 1:
                raise ValueError(
                    f"Several metrics are available for fold {fold}. "
                    f"Please choose which one you want to read among {available_metrics}"
                )
            else:
                selection_metric = available_metrics[0]
        else:
            if selection_metric not in available_metrics:
                raise ValueError(
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
            ValueError: if data_group not in ["train", "validation"] and there is an intersection
                between the participant IDs in test_df and the ones used for training.
        """
        if data_group not in ["train", "validation"]:
            train_path = path.join(self.maps_path, "groups", "train+validation.tsv")
            train_df = pd.read_csv(train_path, sep="\t")
            participants_train = set(train_df.participant_id.values)
            participants_test = set(test_df.participant_id.values)
            intersection = participants_test & participants_train

            if len(intersection) > 0:
                raise ValueError(
                    "Your evaluation set contains participants who were already seen during "
                    "the training step. The list of common participants is the following: "
                    f"{intersection}."
                )

    def _check_data_group(
        self,
        data_group,
        fold,
        caps_directory=None,
        df=None,
        multi_cohort=False,
        overwrite=False,
    ):
        """
        Check if a data group is already available if other arguments are None.
        Else creates a new data_group.

        Args:
            data_group (str): name of the data group
            fold (int): fold number (used to load train and validation folds)
            caps_directory  (str): input CAPS directory
            df (pd.DataFrame): Table of participant_id / session_id of the data group
            multi_cohort (bool): indicates if the input data comes from several CAPS
            overwrite (bool): If True former definition of data group is erased

        Raises:
            ValueError:
                when trying to overwrite train or validation data groups
                when caps_directory or df are given but data group already exists
                when caps_directory or df are not given and data group does not exist
        """
        group_path = path.join(self.maps_path, "groups", data_group)
        logger.debug(f"Group path {group_path}")
        if path.exists(group_path):
            if overwrite:
                if data_group in ["train", "validation"]:
                    raise ValueError("Cannot overwrite train or validation data group.")
                else:
                    shutil.rmtree(group_path)
            elif df is not None or caps_directory is not None:
                raise ValueError(
                    f"Data group {data_group} is already defined. "
                    f"Please do not give any caps_directory, tsv_path or multi_cohort to use it. "
                    f"To erase {data_group} please set overwrite to True."
                )
            else:
                return self.get_group_info(data_group, fold)

        if caps_directory is None or df is None:
            raise ValueError(
                f"The data group {data_group} does not already exist. "
                f"Please specify a caps_directory and a tsv_path to create this data group."
            )
        else:
            self._check_leakage(data_group, df)
            self._write_data_group(data_group, df, caps_directory, multi_cohort)
            return df, {"caps_directory": caps_directory, "multi_cohort": multi_cohort}

    ###############################
    # File writers                #
    ###############################
    @staticmethod
    def write_parameters(json_path, parameters):
        """Write JSON files of parameters."""
        makedirs(json_path, exist_ok=True)

        # save to json file
        json_data = json.dumps(parameters, skipkeys=True, indent=4)
        json_path = path.join(json_path, "maps.json")
        logger.info(f"Path of json file: {json_path}")
        with open(json_path, "w") as f:
            f.write(json_data)

    def _write_requirements_version(self):
        """Writes the environment.txt file."""
        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with open(path.join(self.maps_path, "environment.txt"), "w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_training_data(self):
        """Writes the TSV file containing the participant and session IDs used for training."""
        from clinicadl.utils.caps_dataset.data import load_data_test

        train_df = load_data_test(
            self.tsv_path,
            self.diagnoses,
            baseline=False,
            multi_cohort=self.multi_cohort,
        )
        train_df = train_df[["participant_id", "session_id"]]
        if self.transfer_path:
            transfer_train_path = path.join(
                self.transfer_path, "groups", "train+validation.tsv"
            )
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)

        train_df.to_csv(
            path.join(self.maps_path, "groups", "train+validation.tsv"),
            sep="\t",
            index=False,
        )

    def _write_data_group(
        self,
        data_group,
        df,
        caps_directory=None,
        multi_cohort=None,
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
        group_path = path.join(self.maps_path, "groups", data_group)
        makedirs(group_path)

        columns = ["participant_id", "session_id", "cohort"]
        if self.label in df.columns.values:
            columns += [self.label]

        df.to_csv(path.join(group_path, "data.tsv"), sep="\t", columns=columns)
        self.write_parameters(
            group_path,
            {
                "caps_directory": caps_directory
                if caps_directory is not None
                else self.caps_directory,
                "multi_cohort": multi_cohort
                if multi_cohort is not None
                else self.multi_cohort,
            },
        )

    def _write_train_val_groups(self):
        """Defines the training and validation groups at the initialization"""
        split_manager = self._init_split_manager()
        for fold in split_manager.fold_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[fold][data_group]
                group_path = path.join(
                    self.maps_path, "groups", data_group, f"fold-{fold}"
                )
                makedirs(group_path, exist_ok=True)

                columns = ["participant_id", "session_id", "cohort"]
                if self.label is not None:
                    columns.append(self.label)

                df.to_csv(path.join(group_path, "data.tsv"), sep="\t", columns=columns)
                self.write_parameters(
                    group_path,
                    {
                        "caps_directory": self.caps_directory,
                        "multi_cohort": self.multi_cohort,
                    },
                )

    def _write_weights(
        self,
        state: Dict[str, Any],
        metrics_dict: Optional[Dict[str, bool]],
        fold: int,
        network: int = None,
        filename: str = "checkpoint.pth.tar",
    ):
        """
        Update checkpoint and save the best model according to a set of metrics.
        If no metrics_dict is given, only the checkpoint is saved.

        Args:
            state: state of the training (model weights, epoch...)
            metrics_dict: output of RetainBest step
            fold: fold number
            network: network number (multi-network framework)
            filename: name of the checkpoint file
        """
        checkpoint_dir = path.join(self.maps_path, f"fold-{fold}", "tmp")
        makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = path.join(checkpoint_dir, filename)
        torch.save(state, checkpoint_path)

        best_filename = "model.pth.tar"
        if network is not None:
            best_filename = f"network-{network}_model.pth.tar"

        # Save model according to several metrics
        if metrics_dict is not None:
            for metric_name, metric_bool in metrics_dict.items():
                metric_path = path.join(
                    self.maps_path, f"fold-{fold}", f"best-{metric_name}"
                )
                if metric_bool:
                    makedirs(metric_path, exist_ok=True)
                    shutil.copyfile(
                        checkpoint_path, path.join(metric_path, best_filename)
                    )

    def _erase_tmp(self, fold):
        """Erase checkpoints of the model and optimizer at the end of training."""
        tmp_path = path.join(self.maps_path, f"fold-{fold}", "tmp")
        shutil.rmtree(tmp_path)

    @staticmethod
    def write_description_log(
        log_dir,
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
        makedirs(log_dir, exist_ok=True)
        log_path = path.join(log_dir, "description.log")
        with open(log_path, "w") as f:
            f.write(f"Prediction {data_group} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_dict}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")

    def _mode_level_to_tsv(
        self,
        results_df,
        metrics,
        fold,
        selection,
        data_group="train",
    ):
        """
        Writes the outputs of the test function in tsv files.

        Args:
            results_df: (DataFrame) the individual results per patch.
            metrics: (dict or DataFrame) the performances obtained on a series of metrics.
            fold: (int) the fold for which the performances were obtained.
            selection: (str) the metrics on which the model was selected (BA, loss...)
            data_group: (str) the name referring to the data group on which evaluation is performed.
        """
        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", data_group
        )

        makedirs(performance_dir, exist_ok=True)
        performance_path = path.join(
            performance_dir, f"{data_group}_{self.mode}_level_prediction.tsv"
        )

        if not path.exists(performance_path):
            results_df.to_csv(performance_path, index=False, sep="\t")
        else:
            results_df.to_csv(
                performance_path, index=False, sep="\t", mode="a", header=False
            )

        metrics_path = path.join(
            performance_dir, f"{data_group}_{self.mode}_level_metrics.tsv"
        )
        if metrics is not None:
            if not path.exists(metrics_path):
                pd.DataFrame(metrics, index=[0]).to_csv(
                    metrics_path, index=False, sep="\t"
                )
            else:
                pd.DataFrame(metrics, index=[0]).to_csv(
                    metrics_path, index=False, sep="\t", mode="a", header=False
                )

    def _ensemble_to_tsv(
        self,
        fold,
        selection,
        data_group="test",
        use_labels=True,
    ):
        """
        Writes image-level performance files from mode level performances.

        Args:
            fold: (int) fold number of the cross-validation.
            selection: (str) metric on which the model is selected (for example loss or BA).
            data_group: (str) the name referring to the data group on which evaluation is performed.
                If different from training or validation, the weights of soft voting will be computed
                on validation accuracies.
            use_labels: (bool) If True the labels are added to the final tsv
        """
        # Choose which dataset is used to compute the weights of soft voting.
        if data_group in ["train", "validation"]:
            validation_dataset = data_group
        else:
            validation_dataset = "validation"
        test_df = self.get_prediction(
            data_group, fold, selection, self.mode, verbose=False
        )
        validation_df = self.get_prediction(
            validation_dataset, fold, selection, self.mode, verbose=False
        )

        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", data_group
        )
        makedirs(performance_dir, exist_ok=True)

        df_final, metrics = self.task_manager.ensemble_prediction(
            test_df,
            validation_df,
            selection_threshold=self.selection_threshold,
            use_labels=use_labels,
        )

        if df_final is not None:
            df_final.to_csv(
                path.join(performance_dir, f"{data_group}_image_level_prediction.tsv"),
                index=False,
                sep="\t",
            )
        if metrics is not None:
            pd.DataFrame(metrics, index=[0]).to_csv(
                path.join(performance_dir, f"{data_group}_image_level_metrics.tsv"),
                index=False,
                sep="\t",
            )

    def _mode_to_image_tsv(self, fold, selection, data_group="test", use_labels=True):
        """
        Copy mode-level TSV files to name them as image-level TSV files

        Args:
            fold: (int) Fold number of the cross-validation.
            selection: (str) metric on which the model is selected (for example loss or BA)
            data_group: (str) the name referring to the data group on which evaluation is performed.
            use_labels: (bool) If True the labels are added to the final tsv

        """
        sub_df = self.get_prediction(
            data_group, fold, selection, self.mode, verbose=False
        )
        sub_df.rename(columns={f"{self.mode}_id": "image_id"}, inplace=True)

        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", data_group
        )
        sub_df.to_csv(
            path.join(performance_dir, f"{data_group}_image_level_prediction.tsv"),
            index=False,
            sep="\t",
        )
        if use_labels:
            metrics_df = pd.read_csv(
                path.join(
                    performance_dir, f"{data_group}_{self.mode}_level_metrics.tsv"
                ),
                sep="\t",
            )
            if f"{self.mode}_id" in metrics_df:
                del metrics_df[f"{self.mode}_id"]
            metrics_df.to_csv(
                path.join(performance_dir, f"{data_group}_image_level_metrics.tsv"),
                index=False,
                sep="\t",
            )

    ###############################
    # Objects initialization      #
    ###############################
    def _init_model(
        self,
        transfer_path=None,
        transfer_selection=None,
        fold=None,
        resume=False,
        use_cpu=None,
        network=None,
    ):
        """
        Instantiate the model

        Args:
            transfer_path (str): path to a MAPS in which a model's weights are used for transfer learning.
            transfer_selection (str): name of the metric used to find the source model.
            fold (int): Index of the fold (only used if transfer_path is not None of not resume).
            resume (bool): If True initialize the network with the checkpoint weights.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
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
        if use_cpu is not None:
            kwargs["use_cpu"] = use_cpu

        model = model_class(**kwargs)
        device = model.device
        current_epoch = 0

        if resume:
            checkpoint_path = path.join(
                self.maps_path, f"fold-{fold}", "tmp", "checkpoint.pth.tar"
            )
            checkpoint_state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint_state["model"])
            current_epoch = checkpoint_state["epoch"]
        elif transfer_path:
            logger.debug(f"Transfer weights from MAPS at {transfer_path}")
            transfer_maps = MapsManager(transfer_path)
            transfer_state = transfer_maps.get_state_dict(
                fold,
                selection_metric=transfer_selection,
                network=network,
                map_location=model.device,
            )
            transfer_class = getattr(network_package, transfer_maps.architecture)
            logger.debug(f"Transfer from {transfer_class}")
            model.transfer_weights(transfer_state["model"], transfer_class)

        return model, current_epoch

    def _init_optimizer(self, model, fold=None, resume=False):
        """Initialize the optimizer and use checkpoint weights if resume is True."""
        optimizer = getattr(torch.optim, self.optimizer)(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if resume:
            checkpoint_path = path.join(
                self.maps_path, f"fold-{fold}", "tmp", "optimizer.pth.tar"
            )
            checkpoint_state = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint_state["optimizer"])

        return optimizer

    def _init_split_manager(self, folds=None):
        from clinicadl.utils import split_manager

        split_class = getattr(split_manager, self.validation)
        args = list(
            split_class.__init__.__code__.co_varnames[
                : split_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        args.remove("folds")
        kwargs = {"folds": folds}
        for arg in args:
            kwargs[arg] = self.parameters[arg]
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
            raise ValueError(
                f"Task {self.network_task} is not implemented in ClinicaDL. "
                f"Please choose between classification, regression and reconstruction."
            )

    ###############################
    # Getters                     #
    ###############################
    def _print_description_log(
        self,
        data_group,
        fold,
        selection_metric,
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            data_group (str): name of the data group used for the task.
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
        """
        log_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", data_group
        )
        log_path = path.join(log_dir, "description.log")
        with open(log_path, "r") as f:
            content = f.read()
            print(content)

    def get_group_info(self, data_group, fold=None):
        """
        Gets information from corresponding data group
        (list of participant_id / session_id + configuration parameters).
        fold is only needed if data_group is train or validation.
        """
        group_path = path.join(self.maps_path, "groups", data_group)
        if not path.exists(group_path):
            raise ValueError(
                f"Data group {data_group} is not defined. "
                f"Please run a prediction to create this data group."
            )
        if data_group in ["train", "validation"]:
            if fold is None:
                raise ValueError(
                    f"Information on train or validation data can only be "
                    f"loaded if a fold number is given"
                )
            elif not path.exists(path.join(group_path, f"fold-{fold}")):
                raise ValueError(f"fold {fold} is not available.")
            else:
                group_path = path.join(group_path, f"fold-{fold}")

        df = pd.read_csv(path.join(group_path, "data.tsv"), sep="\t")
        json_path = path.join(group_path, "maps.json")
        with open(json_path, "r") as f:
            parameters = json.load(f)

        return df, parameters

    def get_parameters(self):
        """Returns the training parameters dictionary."""
        json_path = path.join(self.maps_path, "maps.json")
        with open(json_path, "r") as f:
            parameters = json.load(f)

        # Types of retro-compatibility
        # Change arg name: ex network --> model
        # Change arg value: ex for preprocessing: mni --> t1-extensive
        # New arg with default hard-coded value --> discarded_slice --> 20
        retro_change_name = {
            "model": "architecture",
            "multi": "multi_network",
        }
        retro_change_value = {
            # "preprocessing": {"mni": "t1-extensive", "linear": "t1-linear"}
        }
        retro_add = {
            "loss": "default",
        }

        for old_name, new_name in retro_change_name.items():
            if old_name in parameters:
                parameters[new_name] = parameters[old_name]
                del parameters[old_name]

        for name, change_values in retro_change_value.items():
            if parameters[name] in change_values:
                parameters[name] = change_values[parameters[name]]

        for name, value in retro_add.items():
            if name not in parameters:
                parameters[name] = value

        # Build preprocessing_dict
        if "preprocessing_dict" not in parameters:
            parameters["preprocessing_dict"] = {"mode": parameters["mode"]}
            preprocessing_options = [
                "preprocessing",
                "use_uncropped_image",
                "prepare_dl" "custom_suffix",
                "acq_label",
                "suvr_reference_region",
                "patch_size",
                "stride_size",
                "slice_direction",
                "slice_mode",
                "discarded_slices",
                "roi_list",
                "uncropped_roi",
                "roi_custom_suffix",
                "roi_custom_template",
                "roi_custom_mask_pattern",
            ]
            for preprocessing_var in preprocessing_options:
                if preprocessing_var in parameters:
                    parameters["preprocessing_dict"][preprocessing_var] = parameters[
                        preprocessing_var
                    ]
                    del parameters[preprocessing_var]

        # Add missing parameters in previous version of extract
        if "use_uncropped_image" not in parameters["preprocessing_dict"]:
            parameters["preprocessing_dict"]["use_uncropped_image"] = False

        if (
            "prepare_dl" not in parameters["preprocessing_dict"]
            and parameters["mode"] != "image"
        ):
            parameters["preprocessing_dict"]["prepare_dl"] = False

        if (
            parameters["mode"] == "slice"
            and "slice_mode" not in parameters["preprocessing_dict"]
        ):
            parameters["preprocessing_dict"]["slice_mode"] = "rgb"

        if "file_type" not in parameters["preprocessing_dict"]:
            _, file_type = compute_folder_and_file_type(
                parameters["preprocessing_dict"]
            )
            parameters["preprocessing_dict"]["file_type"] = file_type

        return parameters

    def get_model(
        self, fold: int = 0, selection_metric: str = None, network: int = None
    ) -> Network:
        selection_metric = self._check_selection_metric(fold, selection_metric)
        if self.multi_network:
            if network is None:
                raise ValueError(
                    "Please precise the network number that must be loaded."
                )
        return self._init_model(
            self.maps_path, selection_metric, fold, network=network
        )[0]

    def get_state_dict(
        self, fold=0, selection_metric=None, network=None, map_location=None
    ):
        """
        Get the model trained corresponding to one fold and one metric evaluated on the validation set.

        Args:
            fold (int): Index of the fold used for training.
            selection_metric (str): name of the metric used for the selection.
            network (int): Index of the network trained (used in multi-network setting only).
            map_location (str): torch.device object or a string containing a device tag,
                it indicates the location where all tensors should be loaded.
                (see https://pytorch.org/docs/stable/generated/torch.load.html).
        Returns:
            (Dict): dictionary of results (weights, epoch number, metrics values)
        """
        selection_metric = self._check_selection_metric(fold, selection_metric)
        if self.multi_network:
            if network is None:
                raise ValueError(
                    "Please precise the network number that must be loaded."
                )
            else:
                model_path = path.join(
                    self.maps_path,
                    f"fold-{fold}",
                    f"best-{selection_metric}",
                    f"network-{network}_model.pth.tar",
                )
        else:
            model_path = path.join(
                self.maps_path,
                f"fold-{fold}",
                f"best-{selection_metric}",
                "model.pth.tar",
            )

        logger.info(
            f"Loading model trained for fold {fold} "
            f"selected according to best validation {selection_metric} "
            f"at path {model_path}."
        )
        return torch.load(model_path, map_location=map_location)

    def get_prediction(
        self, data_group, fold=0, selection_metric=None, mode="image", verbose=True
    ):
        """
        Get the individual predictions for each participant corresponding to one group
        of participants identified by its data group.

        Args:
            data_group (str): name of the data group used for the prediction task.
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
            mode (str): level of the prediction.
            verbose (bool): if True will print associated prediction.log.
        Returns:
            (DataFrame): Results indexed by columns 'participant_id' and 'session_id' which
            identifies the image in the BIDS / CAPS.
        """
        selection_metric = self._check_selection_metric(fold, selection_metric)
        if verbose:
            self._print_description_log(data_group, fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", data_group
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to data group {data_group} was found."
            )
        df = pd.read_csv(
            path.join(prediction_dir, f"{data_group}_{mode}_level_prediction.tsv"),
            sep="\t",
        )
        df.set_index(["participant_id", "session_id"], inplace=True, drop=True)
        return df

    def get_metrics(
        self, data_group, fold=0, selection_metric=None, mode="image", verbose=True
    ):
        """
        Get the metrics corresponding to a group of participants identified by its data_group.

        Args:
            data_group (str): name of the data group used for the prediction task.
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
            mode (str): level of the prediction
            verbose (bool): if True will print associated prediction.log
        Returns:
            (dict[str:float]): Values of the metrics
        """
        selection_metric = self._check_selection_metric(fold, selection_metric)
        if verbose:
            self._print_description_log(data_group, fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", data_group
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to data group {data_group} was found."
            )
        df = pd.read_csv(
            path.join(prediction_dir, f"{data_group}_{mode}_level_metrics.tsv"),
            sep="\t",
        )
        return df.to_dict("records")[0]

    def get_interpretation(
        self,
        data_group,
        name,
        fold=0,
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
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
            verbose (bool): if True will print associated prediction.log.
            participant_id (str): ID of the participant (if not given load mean map).
            session_id (str): ID of the session (if not give load the mean map).
            mode_id (int): Index of the mode used.
        Returns:
            (torch.Tensor): Tensor of the interpretability map.
        """

        selection_metric = self._check_selection_metric(fold, selection_metric)
        if verbose:
            self._print_description_log(data_group, fold, selection_metric)
        map_dir = path.join(
            self.maps_path,
            f"fold-{fold}",
            f"best-{selection_metric}",
            data_group,
            f"interpret-{name}",
        )
        if not path.exists(map_dir):
            raise ValueError(
                f"No prediction corresponding to data group {data_group} and "
                f"interpretation {name} was found."
            )
        if participant_id is None and session_id is None:
            map_pt = torch.load(
                path.join(map_dir, f"mean_{self.mode}-{mode_id}_map.pt")
            )
        elif participant_id is None or session_id is None:
            raise ValueError(
                f"To load the mean interpretation map, "
                f"please do not give any participant_id or session_id.\n "
                f"Else specify both parameters"
            )
        else:
            map_pt = torch.load(
                path.join(
                    map_dir,
                    f"{participant_id}_{session_id}_{self.mode}-{mode_id}_map.pt",
                )
            )
        return map_pt
