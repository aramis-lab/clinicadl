import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from os import listdir, makedirs, path

import pandas as pd
import torch

from clinicadl.utils.caps_dataset.data import get_transforms, return_dataset
from clinicadl.utils.early_stopping import EarlyStopping
from clinicadl.utils.maps_manager.logwriter import LogWriter, StdLevelFilter
from clinicadl.utils.metric_module import RetainBest

level_dict = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# TODO: replace "fold" with "split"
# TODO save weights on CPU for better compatibility


class MapsManager:
    def __init__(self, maps_path, parameters=None, verbose="warning"):
        """
        Args:
            maps_path (str): path of the MAPS
            parameters (Dict[str:object]): parameters of the training step. If given a new MAPS is created.
            verbose (str): Logging level ("debug", "info", "warning", "error", "critical")
        """
        self.maps_path = maps_path
        self.set_verbose(verbose)

        # Existing MAPS
        if parameters is None:
            if not path.exists(path.join(maps_path, "maps.json")):
                raise ValueError(
                    f"MAPS was not found at {maps_path}."
                    f"To initiate a new MAPS please give a train_dict."
                )
            self.parameters = self.get_parameters()
            self.task_manager = self._init_task_manager()

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
            self.logger.info(f"A new MAPS was created at {maps_path}")
            self._check_args(parameters)
            self._write_parameters()
            self._write_requirements_version()
            self._write_training_data()

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"'MapsManager' object has no attribute '{name}'")

    def set_verbose(self, verbose="warning"):
        """
        Set the verbose to a new level.

        Args:
            verbose (str): Logging level ("debug", "info", "warning", "error", "critical")
        """
        if verbose not in level_dict:
            raise ValueError(
                f"Your verbose value {verbose} is incorrect."
                f"Please choose between the following values {level_dict.keys()}."
            )
        self.logger = logging.getLogger("clinicadl")
        self.logger.setLevel(level_dict[verbose])

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        stdout = logging.StreamHandler(sys.stdout)
        stdout.addFilter(StdLevelFilter())
        stderr = logging.StreamHandler(sys.stderr)
        stderr.addFilter(StdLevelFilter(err=True))
        # create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        # add formatter to ch
        stdout.setFormatter(formatter)
        stderr.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(stdout)
        self.logger.addHandler(stderr)
        self.logger.propagate = False

    def train(self, folds=None, overwrite=False):
        """
        Performs the training task for a defined list of folds

        Args:
            folds (List[int]): list of folds on which the training task is performed.
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
        if self.multi:
            self._train_multi(folds, resume=False)
        else:
            self._train_single(folds, resume=False)

    def resume(self, folds=None):
        """
        Resumes the training task for a defined list of folds

        Args:
            folds (List[int]): list of folds on which the training task is performed.
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

        if self.multi:
            self._train_multi(folds, resume=True)
        else:
            self._train_single(folds, resume=True)

    def predict(
        self,
        caps_directory,
        tsv_path,
        prefix,
        folds=None,
        selection_metrics=None,
        multi_cohort=False,
        preprocessing=None,
        diagnoses=None,
        use_labels=True,
        prepare_dl=None,
        batch_size=None,
        num_workers=None,
        use_cpu=None,
        overwrite=False,
    ):
        """
        Performs the prediction task on a subset of caps_directory defined in a TSV file.

        Args:
            caps_directory (str): path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
            tsv_path (str): path to a TSV file containing the list of participants and sessions to test.
            prefix (str): name of the data group tested.
            folds (List[int]): list of folds to test. Default perform prediction on all folds available.
            selection_metrics (List[str]): list of selection metrics to test.
                Default performs the prediction on all selection metrics available.
            multi_cohort (bool): If True considers that tsv_path is the path to a multi-cohort TSV.
            preprocessing (str): Name of the preprocessing used. Default uses the same as in training step.
            diagnoses (List[str]): List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            prepare_dl (bool): If given, sets the value of prepare_dl, else use the same as in training step.
            batch_size (int): If given, sets the value of batch_size, else use the same as in training step.
            num_workers (int): If given, sets the value of num_workers, else use the same as in training step.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            overwrite (bool): If True erase the occurrences of prefix.
        Raises:
            ValueError: If the predictions with prefix name already exist and overwrite is False.
        """
        from torch.utils.data import DataLoader

        from clinicadl.utils.caps_dataset.data import load_data_test

        if folds is None:
            folds = self._find_folds()

        self._check_prefix(
            prefix,
            folds=folds,
            selection_metrics=selection_metrics,
            overwrite=overwrite,
        )

        _, all_transforms = get_transforms(
            self.mode,
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        test_df = load_data_test(
            tsv_path,
            diagnoses if diagnoses is not None else self.diagnoses,
            multi_cohort=multi_cohort,
        )

        self._check_leakage(test_df)

        criterion = self.task_manager.get_criterion()

        for fold in folds:
            if self.multi:
                for network in range(self.num_networks):
                    data_test = return_dataset(
                        self.mode,
                        caps_directory,
                        test_df,
                        preprocessing
                        if preprocessing is not None
                        else self.preprocessing,
                        all_transformations=all_transforms,
                        prepare_dl=prepare_dl
                        if prepare_dl is not None
                        else self.prepare_dl,
                        multi_cohort=multi_cohort,
                        label_presence=use_labels,
                        label=self.label,
                        label_code=self.label_code,
                        params=self,
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
                        prefix,
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
                    self.mode,
                    caps_directory,
                    test_df,
                    preprocessing if preprocessing is not None else self.preprocessing,
                    all_transformations=all_transforms,
                    prepare_dl=prepare_dl
                    if prepare_dl is not None
                    else self.prepare_dl,
                    multi_cohort=multi_cohort,
                    label_presence=use_labels,
                    label=self.label,
                    label_code=self.label_code,
                    params=self,
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
                    prefix,
                    fold,
                    selection_metrics
                    if selection_metrics is not None
                    else self._find_selection_metrics(fold),
                    use_labels=use_labels,
                    use_cpu=use_cpu,
                )
            self._ensemble_prediction(prefix, fold, selection_metrics, use_labels)

    def interpret(
        self,
        prefix,
        caps_directory=None,
        tsv_path=None,
        folds=None,
        selection_metrics=None,
        multi_cohort=False,
        preprocessing=None,
        diagnoses=None,
        baseline=False,
        target_node=0,
        save_individual=False,
        prepare_dl=None,
        batch_size=None,
        num_workers=None,
        use_cpu=None,
        overwrite=False,
    ):
        """
        Performs the interpretation task on a subset of caps_directory defined in a TSV file.
        The mean interpretation is always saved, to save the individual interpretations set save_individual to True.

        Args:
            caps_directory (str): path to the CAPS folder. For more information please refer to
                [clinica documentation](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/).
                Default use caps_directory of the training step.
            tsv_path (str): path to a TSV file containing the list of participants and sessions to interpret.
                Default use tsv_path of the training step.
            prefix (str): name of the data group interpreted.
            folds (List[int]): list of folds to interpret. Default perform interpretation on all folds available.
            selection_metrics (List[str]): list of selection metrics to interpret.
                Default performs the interpretation on all selection metrics available.
            multi_cohort (bool): If True considers that tsv_path is the path to a multi-cohort TSV.
            preprocessing (str): Name of the preprocessing used. Default uses the same as in training step.
            diagnoses (List[str]): List of diagnoses to load if tsv_path is a split_directory.
                Default uses the same as in training step.
            baseline (bool): If True baseline sessions only are used for interpretation.
            target_node (int): Node from which the interpretation is computed.
            save_individual (bool): If True saves the individual map of each participant / session couple.
            prepare_dl (bool): If given, sets the value of prepare_dl, else use the same as in training step.
            batch_size (bool): If given, sets the value of batch_size, else use the same as in training step.
            num_workers (int): If given, sets the value of num_workers, else use the same as in training step.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            overwrite (bool): If True erase the occurrences of prefix.
        Raises:
            ValueError: If the predictions with prefix name already exist and overwrite is False.
        """

        from torch.utils.data import DataLoader

        from clinicadl.interpret.gradients import VanillaBackProp
        from clinicadl.utils.caps_dataset.data import load_data_test

        if folds is None:
            folds = self._find_folds()

        if self.multi:
            raise NotImplementedError(
                "The interpretation of multi-network framework is not implemented."
            )
        self._check_prefix(
            prefix, folds, selection_metrics, overwrite, interpretation=True
        )

        _, all_transforms = get_transforms(
            self.mode,
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        test_df = load_data_test(
            tsv_path if tsv_path is not None else self.tsv_path,
            diagnoses if diagnoses is not None else self.diagnoses,
            multi_cohort=multi_cohort,
            baseline=baseline,
        )
        data_test = return_dataset(
            self.mode,
            caps_directory if caps_directory is not None else self.caps_directory,
            test_df,
            preprocessing if preprocessing is not None else self.preprocessing,
            all_transformations=all_transforms,
            prepare_dl=prepare_dl if prepare_dl is not None else self.prepare_dl,
            multi_cohort=multi_cohort,
            label_presence=False,
            label_code=self.label_code,
            label=self.label,
            params=self,
        )
        test_loader = DataLoader(
            data_test,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else self.num_workers,
        )

        for fold in folds:
            self.logger.info(f"Interpretation of fold {fold}")
            if selection_metrics is None:
                selection_metrics = self._find_selection_metrics(fold)

            for selection_metric in selection_metrics:
                self.logger.info(f"Interpretation of metric {selection_metric}")
                self._write_description_log(
                    prefix,
                    fold,
                    selection_metric,
                    data_test.caps_dict,
                    data_test.df,
                    interpretation=True,
                    params_dict={
                        "target_node": target_node,
                        "preprocessing": preprocessing,
                        "diagnoses": diagnoses,
                    },
                )
                results_path = path.join(
                    self.maps_path,
                    f"fold-{fold}",
                    f"best-{selection_metric}",
                    "interpretation",
                    prefix,
                )

                model, _ = self._init_model(
                    transfer_path=self.maps_path,
                    fold=fold,
                    transfer_selection=selection_metric,
                    use_cpu=use_cpu,
                )

                interpreter = VanillaBackProp(model)

                mean_map = 0
                for data in test_loader:
                    images = data["image"].to(model.device)

                    map_pt = interpreter.generate_gradients(images, target_node)
                    mean_map += map_pt.sum(axis=0)
                    if save_individual:
                        for i in range(len(data["participant_id"])):
                            single_path = path.join(
                                results_path,
                                f"participant-{data['participant_id'][i]}_session-{data['session_id'][i]}_"
                                f"{self.mode}-{data[f'{self.mode}_id'][i]}_map.pt",
                            )
                            torch.save(map_pt, single_path)
                mean_map /= len(data_test)
                torch.save(mean_map, path.join(results_path, f"mean_map.pt"))

    ###################################
    # High-level functions templates  #
    ###################################
    def _train_single(self, folds=None, resume=False):
        """
        Trains a single CNN for all inputs.

        Args:
            folds (List[int]): list of folds that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        from torch.utils.data import DataLoader

        train_transforms, all_transforms = get_transforms(
            self.mode,
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            self.logger.info(f"Training fold {fold}")

            fold_df_dict = split_manager[fold]

            data_train = return_dataset(
                self.mode,
                self.caps_directory,
                fold_df_dict["train"],
                self.preprocessing,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                prepare_dl=self.prepare_dl,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
                params=self,
            )
            data_valid = return_dataset(
                self.mode,
                self.caps_directory,
                fold_df_dict["validation"],
                self.preprocessing,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                prepare_dl=self.prepare_dl,
                multi_cohort=self.multi_cohort,
                label=self.label,
                label_code=self.label_code,
                params=self,
            )

            train_sampler = self.task_manager.generate_sampler(data_train, self.sampler)

            train_loader = DataLoader(
                data_train,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
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
            folds (List[int]): list of folds that are trained.
            resume (bool): If True the job is resumed from checkpoint.
        """
        from torch.utils.data import DataLoader

        train_transforms, all_transforms = get_transforms(
            self.mode,
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            self.logger.info(f"Training fold {fold}")

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
                self.logger.info(f"Train network {network}")

                data_train = return_dataset(
                    self.mode,
                    self.caps_directory,
                    fold_df_dict["train"],
                    self.preprocessing,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    prepare_dl=self.prepare_dl,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                    cnn_index=network,
                    params=self,
                )
                data_valid = return_dataset(
                    self.mode,
                    self.caps_directory,
                    fold_df_dict["validation"],
                    self.preprocessing,
                    train_transformations=train_transforms,
                    all_transformations=all_transforms,
                    prepare_dl=self.prepare_dl,
                    multi_cohort=self.multi_cohort,
                    label=self.label,
                    label_code=self.label_code,
                    cnn_index=network,
                    params=self,
                )

                train_sampler = self.task_manager.generate_sampler(
                    data_train, self.sampler
                )

                train_loader = DataLoader(
                    data_train,
                    batch_size=self.batch_size,
                    sampler=train_sampler,
                    num_workers=self.num_workers,
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

        model, beginning_epoch = self._init_model(fold=fold, resume=resume)
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
            self.task_manager.evaluation_metrics,
            fold,
            resume=resume,
            beginning_epoch=beginning_epoch,
            network=network,
        )
        epoch = log_writer.beginning_epoch

        retain_best = RetainBest(selection_metrics=self.selection_metrics)

        while epoch < self.epochs and not early_stopping.step(metrics_valid["loss"]):
            self.logger.info(f"Beginning epoch {epoch}.")

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
                        self.logger.info(
                            f"{self.mode} level training loss is {metrics_train['loss']} "
                            f"at the end of iteration {i}"
                        )
                        self.logger.info(
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
                self.logger.warning(
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
            self.logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

            _, metrics_train = self.task_manager.test(model, train_loader, criterion)
            _, metrics_valid = self.task_manager.test(model, valid_loader, criterion)

            model.train()
            train_loader.dataset.train()

            log_writer.step(epoch, i, metrics_train, metrics_valid, len(train_loader))
            self.logger.info(
                f"{self.mode} level training loss is {metrics_train['loss']} "
                f"at the end of iteration {i}"
            )
            self.logger.info(
                f"{self.mode} level validation loss is {metrics_valid['loss']} "
                f"at the end of iteration {i}"
            )

            # Save checkpoints and best models
            best_dict = retain_best.step(metrics_valid)
            self._write_weights(
                {"model": model.state_dict(), "epoch": epoch, "name": self.architecture},
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

    def _test_loader(
        self,
        dataloader,
        criterion,
        prefix,
        fold,
        selection_metrics,
        use_labels=True,
        use_cpu=None,
        network=None,
    ):
        """
        Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test set.
            criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
            prefix (str): name of the data group used for the testing task.
            fold (int): Index of the fold used to train the model tested.
            selection_metrics (List[str]): List of metrics used to select the best models which are tested.
            use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
            use_cpu (bool): If given, a new value for the device of the model will be computed.
            network (int): Index of the network tested (only used in multi-network setting).
        """
        for selection_metric in selection_metrics:

            self._write_description_log(
                prefix,
                fold,
                selection_metric,
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
                self.logger.info(
                    f"{self.mode} level {prefix} loss is {metrics['loss']} for model selected on {selection_metric}"
                )

            # Replace here
            self._mode_level_to_tsv(
                prediction_df, metrics, fold, selection_metric, prefix=prefix
            )

    def _ensemble_prediction(
        self,
        prefix,
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
                    prefix=prefix,
                    use_labels=use_labels,
                )
            elif self.mode != "image":
                self._mode_to_image_tsv(
                    fold,
                    selection=selection_metric,
                    prefix=prefix,
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
            "preprocessing",
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

        train_parameters = self._compute_train_args()
        self.parameters.update(train_parameters)

        if self.parameters["num_networks"] < 2 and self.multi:
            raise ValueError(
                f"Invalid training arguments: cannot train a multi-network "
                f"framework with only {self.parameters['num_networks']} element "
                f"per image."
            )

        # TODO: add default values manager
        # click passing context @click.command / @click.passcontext (config.json)
        # or default parameters in click --> from config_param import learning_rate --> @learning_rate

    def _compute_train_args(self):

        if "label" not in self.parameters:
            self.parameters["label"] = None
        if "selection_threshold" not in self.parameters:
            self.parameters["selection_threshold"] = None

        _, transformations = get_transforms(self.mode, self.minmaxnormalization)

        split_manager = self._init_split_manager(None)
        train_df = split_manager[0]["train"]
        self.task_manager = self._init_task_manager()
        label_code = self.task_manager.generate_label_code(train_df, self.label)
        full_dataset = return_dataset(
            self.mode,
            self.caps_directory,
            train_df,
            self.preprocessing,
            label=self.label,
            label_code=label_code,
            train_transformations=None,
            all_transformations=transformations,
            params=self,
        )

        return {
            "num_networks": full_dataset.elem_per_image,
            "label_code": label_code,
            "output_size": self.task_manager.output_size(
                full_dataset.size, full_dataset.df, self.label
            ),
            "input_size": full_dataset.size,
        }

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

    def _check_prefix(
        self,
        prefix,
        folds=None,
        selection_metrics=None,
        overwrite=False,
        interpretation=False,
    ):
        """
        Check that a prefix is available for a list of folds and selection metrics.

        Args:
            prefix (str): name whose presence is checked.
            folds (List[int]): list of folds checked. Default checks all folds available.
            selection_metrics (List[str]): list of selection metrics checked.
                Default checks all selection metrics available.
            overwrite (bool): If True erase the occurrences of prefix.
            interpretation (bool): If True looks for interpretation prefix, else test prefix.
        Raises:
            ValueError: if an occurrence of prefix is found and overwrite is set to False.
        """
        already_evaluated = []
        split_manager = self._init_split_manager(folds)
        for fold in split_manager.fold_iterator():
            if selection_metrics is None:
                selection_metrics = self._find_selection_metrics(fold)
            for selection_metric in selection_metrics:
                self._check_selection_metric(fold, selection_metric)
                if interpretation:
                    prediction_dir = path.join(
                        self.maps_path,
                        f"fold-{fold}",
                        f"best-{selection_metric}",
                        "interpretation",
                        prefix,
                    )
                else:
                    prediction_dir = path.join(
                        self.maps_path,
                        f"fold-{fold}",
                        f"best-{selection_metric}",
                        prefix,
                    )
                if path.exists(prediction_dir):
                    if overwrite:
                        shutil.rmtree(prediction_dir)
                    else:
                        already_evaluated.append(
                            f"- fold {fold}, selection_metric {selection_metric}\n"
                        )

        if len(already_evaluated) > 0:
            error_message = (
                f"Evaluations corresponding to prefix {prefix} were found. \n"
                f"Please use overwrite to erase the evaluations previously done. \n"
                f"List of evaluations already performed:\n"
            )
            for message in already_evaluated:
                error_message += message
            raise ValueError(error_message)

    def _check_leakage(self, test_df):
        """Checks that no intersection exist between the participants used for training and those used for testing."""
        train_path = path.join(self.maps_path, "train_data.tsv")
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

    ###############################
    # File writers                #
    ###############################
    def _write_parameters(self):
        """Write the JSON file of parameters."""
        makedirs(self.maps_path, exist_ok=True)

        # save to json file
        json_data = json.dumps(self.parameters, skipkeys=True, indent=4)
        json_path = path.join(self.maps_path, "maps.json")
        self.logger.info(f"Path of json file: {json_path}")
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
            self.logger.warning(
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
        if self.transfer_path!="":
            transfer_train_path = path.join(self.transfer_path, "train_data.tsv")
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)

        train_df.to_csv(
            path.join(self.maps_path, "train_data.tsv"), sep="\t", index=False
        )

    def _write_weights(
        self, state, metrics_dict, fold, network=None, filename="checkpoint.pth.tar"
    ):
        """
        Update checkpoint and save the best model according to a set of metrics.
        If no metrics_dict is given, only the checkpoint is saved.

        Args:
            state: (Dict[str,object]) state of the training (model weights, epoch...)
            metrics_dict: (Dict[str,bool]) output of RetainBest step
            fold: (int) fold number
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

    def _write_description_log(
        self,
        prefix,
        fold,
        selection_metric,
        caps_directory,
        df,
        interpretation=False,
        params_dict=None,
    ):
        """
        Write description log associated to predict or interpret task.

        Args:
            prefix (str): name of the data group used for the task.
            fold (int): Index of the fold used for training.
            selection_metric (str): selection metric used to select the best model.
            caps_directory (str): CAPS used for the task
            df (pd.DataFrame): DataFrame of the meta-data used for the task.
            interpretation (bool): If True looks for interpretation prefix, else test prefix.
            params_dict (Dict[str, Any]): set of parameters used for the task.
        """
        if interpretation:
            log_dir = path.join(
                self.maps_path,
                f"fold-{fold}",
                f"best-{selection_metric}",
                "interpretation",
                prefix,
            )
        else:
            log_dir = path.join(
                self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
            )
        makedirs(log_dir, exist_ok=True)
        log_path = path.join(log_dir, "description.log")
        with open(log_path, "w") as f:
            f.write(f"Evaluation of {prefix} group - {datetime.now()}\n")
            f.write(f"Data loaded from CAPS directories: {caps_directory}\n")
            f.write(f"Number of participants: {df.participant_id.nunique()}\n")
            f.write(f"Number of sessions: {len(df)}\n")
            if params_dict:
                f.write(f"Other parameters: {params_dict}")

    def _mode_level_to_tsv(
        self,
        results_df,
        metrics,
        fold,
        selection,
        prefix="train",
    ):
        """
        Writes the outputs of the test function in tsv files.

        Args:
            results_df: (DataFrame) the individual results per patch.
            metrics: (dict or DataFrame) the performances obtained on a series of metrics.
            fold: (int) the fold for which the performances were obtained.
            selection: (str) the metrics on which the model was selected (BA, loss...)
            prefix: (str) the prefix referring to the data group on which evaluation is performed.
        """
        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", prefix
        )

        makedirs(performance_dir, exist_ok=True)
        performance_path = path.join(
            performance_dir, f"{prefix}_{self.mode}_level_prediction.tsv"
        )

        if not path.exists(performance_path):
            results_df.to_csv(performance_path, index=False, sep="\t")
        else:
            results_df.to_csv(
                performance_path, index=False, sep="\t", mode="a", header=False
            )

        metrics_path = path.join(
            performance_dir, f"{prefix}_{self.mode}_level_metrics.tsv"
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
        prefix="test",
        use_labels=True,
    ):
        """
        Writes image-level performance files from mode level performances.

        Args:
            fold: (int) fold number of the cross-validation.
            selection: (str) metric on which the model is selected (for example loss or BA).
            prefix: (str) the prefix referring to the data group on which evaluation is performed.
                If different from training or validation, the weights of soft voting will be computed
                on validation accuracies.
            use_labels: (bool) If True the labels are added to the final tsv
        """
        # Choose which dataset is used to compute the weights of soft voting.
        if prefix in ["train", "validation"]:
            validation_dataset = prefix
        else:
            validation_dataset = "validation"
        test_df = self.get_prediction(prefix, fold, selection, self.mode, verbose=False)
        validation_df = self.get_prediction(
            validation_dataset, fold, selection, self.mode, verbose=False
        )

        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", prefix
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
                path.join(performance_dir, f"{prefix}_image_level_prediction.tsv"),
                index=False,
                sep="\t",
            )
        if metrics is not None:
            pd.DataFrame(metrics, index=[0]).to_csv(
                path.join(performance_dir, f"{prefix}_image_level_metrics.tsv"),
                index=False,
                sep="\t",
            )

    def _mode_to_image_tsv(self, fold, selection, prefix="test", use_labels=True):
        """
        Copy mode-level TSV files to name them as image-level TSV files

        Args:
            fold: (int) Fold number of the cross-validation.
            selection: (str) metric on which the model is selected (for example loss or BA)
            prefix: (str) the prefix referring to the data group on which evaluation is performed.
            use_labels: (bool) If True the labels are added to the final tsv

        """
        sub_df = self.get_prediction(prefix, fold, selection, self.mode, verbose=False)
        sub_df.rename(columns={f"{self.mode}_id": "image_id"}, inplace=True)

        performance_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection}", prefix
        )
        sub_df.to_csv(
            path.join(performance_dir, f"{prefix}_image_level_prediction.tsv"),
            index=False,
            sep="\t",
        )
        if use_labels:
            metrics_df = pd.read_csv(
                path.join(performance_dir, f"{prefix}_{self.mode}_level_metrics.tsv"),
                sep="\t",
            )
            if f"{self.mode}_id" in metrics_df:
                del metrics_df[f"{self.mode}_id"]
            metrics_df.to_csv(
                path.join(performance_dir, f"{prefix}_image_level_metrics.tsv"),
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

        self.logger.debug(f"Initialization of model {self.architecture}")
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
        elif transfer_path is not None:
            self.logger.debug(f"Transfer weights from MAPS at {transfer_path}")
            transfer_maps = MapsManager(transfer_path)
            transfer_state = transfer_maps.get_state_dict(
                fold,
                selection_metric=transfer_selection,
                network=network,
                map_location=model.device,
            )
            transfer_class = getattr(network_package, transfer_maps.model)
            self.logger.debug(f"Transfer from {transfer_class}")
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
        args.remove("logger")
        kwargs = {"folds": folds, "logger": self.logger}
        for arg in args:
            kwargs[arg] = self.parameters[arg]
        return split_class(**kwargs)

    def _init_task_manager(self):
        from clinicadl.utils.task_manager import (
            ClassificationManager,
            ReconstructionManager,
            RegressionManager,
        )

        if self.network_task == "classification":
            return ClassificationManager(self.mode)
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
        self, prefix, fold, selection_metric, interpretation=False
    ):
        """
        Print the description log associated to a prediction or interpretation.

        Args:
            prefix (str): name of the data group used for the task.
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
            interpretation (bool): If True looks for interpretation prefix, else test prefix.
        """
        if interpretation:
            log_dir = path.join(
                self.maps_path,
                f"fold-{fold}",
                f"best-{selection_metric}",
                "interpretation",
                prefix,
            )
        else:
            log_dir = path.join(
                self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
            )
        log_path = path.join(log_dir, "description.log")
        with open(log_path, "r") as f:
            content = f.read()
            print(content)

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
            "network": "model",
            "pretrained_path": "transfer_learning_path",
            "pretrained_difference": "transfer_learning_difference",
            "patch_stride": "stride_size",
            "selection": "transfer_learning_selection",
        }
        retro_change_value = {
            "preprocessing": {"mni": "t1-extensive", "linear": "t1-linear"}
        }
        retro_add = {
            "discarded_slices": 20,
            "loss": "default",
            "uncropped_roi": False,
            "roi_list": None,
            "multi_cohort": False,
            "predict_atlas_intensities": None,  # To remove after multi-task implementation
            "merged_tsv_path": None,  # To remove after multi-task implementation
            "atlas_weight": 1,  # To remove after multi-task implementation
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

        return parameters

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
        if self.multi:
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

        self.logger.info(
            f"Loading model trained for fold {fold} "
            f"selected according to best validation {selection_metric} "
            f"at path {model_path}."
        )
        return torch.load(model_path, map_location=map_location)

    def get_prediction(
        self, prefix, fold=0, selection_metric=None, mode="image", verbose=True
    ):
        """
        Get the individual predictions for each participant corresponding to one group
        of participants identified by its prefix.

        Args:
            prefix (str): name of the data group used for the prediction task.
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
            self._print_description_log(prefix, fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to prefix {prefix} was found."
            )
        df = pd.read_csv(
            path.join(prediction_dir, f"{prefix}_{mode}_level_prediction.tsv"), sep="\t"
        )
        df.set_index(["participant_id", "session_id"], inplace=True, drop=True)
        return df

    def get_metrics(
        self, prefix, fold=0, selection_metric=None, mode="image", verbose=True
    ):
        """
        Get the metrics corresponding to a group of participants identified by its prefix.

        Args:
            prefix (str): name of the data group used for the prediction task.
            fold (int): Index of the fold used for training.
            selection_metric (str): Metric used for best weights selection.
            mode (str): level of the prediction
            verbose (bool): if True will print associated prediction.log
        Returns:
            (Dict[str:float]): Values of the metrics
        """
        selection_metric = self._check_selection_metric(fold, selection_metric)
        if verbose:
            self._print_description_log(prefix, fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to prefix {prefix} was found."
            )
        df = pd.read_csv(
            path.join(prediction_dir, f"{prefix}_{mode}_level_metrics.tsv"), sep="\t"
        )
        return df.to_dict("records")[0]

    def get_interpretation(
        self,
        prefix,
        fold=0,
        selection_metric=None,
        verbose=True,
        participant_id=None,
        session_id=None,
        mode_id=0,
    ):
        """
        Get the individual interpretation maps for one session if participant_id and session_id are filled.
        Else load the mean interpretation map.

        Args:
            prefix (str): Name of the data group used for the interpretation task.
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
            self._print_description_log(
                prefix, fold, selection_metric, interpretation=True
            )
        map_dir = path.join(
            self.maps_path,
            f"fold-{fold}",
            f"best-{selection_metric}",
            "interpretation",
            prefix,
        )
        if not path.exists(map_dir):
            raise ValueError(
                f"No prediction corresponding to prefix {prefix} was found."
            )
        if participant_id is None and session_id is None:
            map_pt = torch.load(path.join(map_dir, "mean_map.pt"))
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
                    f"participant-{participant_id}_session-{session_id}_{self.mode}-{mode_id}_map.pt",
                )
            )
        return map_pt
