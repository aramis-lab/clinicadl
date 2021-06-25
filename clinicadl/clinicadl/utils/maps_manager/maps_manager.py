import logging

from clinicadl.utils.maps_manager import LogWriter
from clinicadl.utils.metric_module import MetricModule, RetainBest

# Options
# selection_metrics: balanced_accuracy, loss

# TODO clarify difference between evaluation_metrics and selection_metrics

level_dict = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# TODO: replace "fold" with "split"


class MapsManager:
    def __init__(self, maps_path, parameters=None, verbose="warning"):
        """
        Args:
            maps_path (str): path of the MAPS
            parameters (Dict[str:object]): parameters of the training step. If given a new MAPS is created.
            verbose (str): Logging level ("debug", "info", "warning", "error", "critical")
        """
        from os import listdir, path

        self.maps_path = maps_path
        self.set_verbose(verbose)

        # Existing MAPS
        if parameters is None:
            if not path.exists(path.join(maps_path, "maps.json")):
                raise ValueError(
                    f"MAPS was not found at {maps_path}."
                    f"To initiate a new MAPS please give a train_dict."
                )
            self.parameters = self._load_parameters()

        # Initiate MAPS
        else:
            if (
                path.exists(maps_path) and not path.isdir(maps_path)
            ) or (  # Non-folder file
                path.isdir(maps_path) and listdir(maps_path)  # Non empty folder
            ):
                raise ValueError(
                    f"You are trying a new MAPS at {maps_path} but"
                    f"this already corresponds to a file or a non-empty folder."
                    f"Please remove it or choose another location."
                )
            self.logger.info(f"A new MAPS was created at {maps_path}")
            self._check_args(parameters)
            self.parameters = parameters
            self._write_parameters()
            self._write_requirements_version()

        self.log_writer = LogWriter(self.maps_path, self.evaluation_metrics)

    def _load_parameters(self):
        import json
        from os import path

        json_path = path.join(self.maps_path, "maps.json")
        with open(json_path, "r") as f:
            parameters = json.load(f)

        # Types of retro-compatibility
        # Change arg name: ex network --> model
        # Change arg value: ex for preprocessing: mni --> t1-extensive
        # New arg with default hard-coded value --> discarded_slice --> 20
        retro_change_name = {
            "network": "model",
            "mri_plane": "slice_direction",
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

    @staticmethod
    def _check_args(parameters):
        mandatory_arguments = [
            "caps_directory",
            "tsv_path",
            "preprocessing",
            "mode",
            "network_type",
            "model",
        ]

        for arg in mandatory_arguments:
            if arg not in parameters:
                raise ValueError(
                    f"The values of mandatory arguments {mandatory_arguments} should be set."
                    f"No value was given for {arg}."
                )

        # TODO: add default values manager
        # click passing context @click.command / @click.passcontext (config.json)
        # or default parameters in click --> from config_param import learning_rate --> @learning_rate

    def _write_parameters(self):
        import json
        from os import makedirs, path

        makedirs(self.maps_path, exist_ok=True)

        # save to json file
        json = json.dumps(self.parameters, skipkeys=True, indent=4)
        json_path = path.join(self.maps_path, "maps.json")
        self.logger.info(f"Path of json file: {json_path}")
        with open(json_path, "w") as f:
            f.write(json)

    def _write_requirements_version(self):
        import subprocess
        from os import path

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

    def _write_weights(self, state, metrics_dict, fold, filename="checkpoint.pth.tar"):
        """
        Update checkpoint and save the best model according to a set of metrics.
        If no metrics_dict is given, only the checkpoint is saved.

        Args:
            state: (Dict[str,object]) state of the training (model weights, epoch...)
            metrics_dict: (Dict[str,bool]) output of RetainBest step
            fold: (int) fold number
        """
        import os
        from os import path

        import nibabel as nib
        import torch

        checkpoint_path = path.join(self.maps_path, f"fold-{fold}", "tmp", filename)
        torch.save(state, checkpoint_path)

        # Save model according to several metrics
        if metrics_dict is not None:
            for metric_name, metric_bool in metrics_dict.items():
                metric_path = path.join(
                    self.maps_path, f"fold-{fold}", f"best_{metric_name}"
                )
                if metric_bool:
                    os.makedirs(metric_path, exist_ok=True)
                    shutil.copyfile(
                        checkpoint_path, path.join(metric_path, "model.pth.tar")
                    )

    def _erase_tmp(self, fold):
        import os

        tmp_path = path.join(self.maps_path, f"fold-{fold}", "tmp")
        os.remove(tmp_path)

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"'MapsManager' object has no attribute '{name}'")

    def set_verbose(self, verbose="warning"):
        """
        Args:
            verbose (str): Logging level ("debug", "info", "warning", "error", "critical")
        """
        import logging
        import sys
        from logging import getLogger

        from clinicadl.utils.maps_manager.logwriter import StdLevelFilter

        if verbose not in level_dict:
            raise ValueError(
                f"Your verbose value {verbose} is incorrect."
                f"Please choose between the following values {level_dict.keys()}."
            )
        self.logger = getLogger("clinicadl")
        self.logger.setLevel(level_dict[verbose])

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

    def train(self, folds=None):
        """
        Args:
            folds (List[int]): list of folds that are trained
        """
        from time import time

        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter

        from clinicadl.utils.caps_dataset.data import (
            generate_sampler,
            get_transforms,
            load_data,
            return_dataset,
        )
        from clinicadl.utils.early_stopping import EarlyStopping

        train_transforms, all_transforms = get_transforms(
            self.mode,
            minmaxnormalization=self.minmaxnormalization,
            data_augmentation=self.data_augmentation,
        )

        # TODO: check folds do not exist yet
        # TODO: replace by CV
        if folds is None:
            if self.n_splits is None:
                fold_iterator = range(1)
            else:
                fold_iterator = range(self.n_splits)
        else:
            fold_iterator = folds

        for fold in fold_iterator:
            self.logger.info(f"Training fold {fold}")

            # TODO generate DataFrames with CV --> specify here multi_cohort
            training_df, valid_df = load_data(
                self.tsv_path,
                self.diagnoses,
                fold,
                n_splits=self.n_splits,
                baseline=self.baseline,
                logger=self.logger,
                multi_cohort=self.multi_cohort,
            )

            data_train = return_dataset(
                self.mode,
                self.caps_directory,
                training_df,
                self.preprocessing,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                prepare_dl=self.prepare_dl,
                multi_cohort=self.multi_cohort,
                params=self,
            )
            data_valid = return_dataset(
                self.mode,
                self.caps_directory,
                valid_df,
                self.preprocessing,
                train_transformations=train_transforms,
                all_transformations=all_transforms,
                prepare_dl=self.prepare_dl,
                multi_cohort=self.multi_cohort,
                params=self,
            )

            train_sampler = generate_sampler(data_train, self.sampler)

            train_loader = DataLoader(
                data_train,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            valid_loader = DataLoader(
                data_valid,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            model = self._init_model(input_shape=data_train.size)
            model.train()
            train_loader.dataset.train()
            criterion = self._get_criterion()
            optimizer = self._init_optimizer(model)

            retain_best = RetainBest(selection_list=self.selection_metrics)

            early_stopping = EarlyStopping(
                "min", min_delta=self.tolerance, patience=self.patience
            )
            metrics_valid = {"total_loss": None}

            self.log_writer.init_fold(fold)
            epoch = self.log_writer.beginning_epoch

            while epoch < self.epochs and not early_stopping.step(
                metrics_valid["total_loss"]
            ):
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

                            _, metrics_train = test(
                                model, train_loader, options.gpu, criterion
                            )
                            _, metrics_valid = test(
                                model, valid_loader, options.gpu, criterion
                            )

                            model.train()
                            train_loader.dataset.train()

                            self.log_writer.step(
                                fold,
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
                                f"{self.mode} level validation loss is {metrics_train['loss']} "
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
                if (i + 1) % options.accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # Always test the results and save them once at the end of the epoch
                model.zero_grad()
                self.logger.debug(f"Last checkpoint at the end of the epoch {epoch}")

                _, metrics_train = test(model, train_loader, options.gpu, criterion)
                _, metrics_valid = test(model, valid_loader, options.gpu, criterion)

                model.train()
                train_loader.dataset.train()

                self.log_writer(
                    fold, epoch, i, metrics_train, metrics_valid, len(train_loader)
                )
                self.logger.info(
                    f"{self.mode} level training loss is {metrics_train['loss']} "
                    f"at the end of iteration {i}"
                )
                self.logger.info(
                    f"{self.mode} level validation loss is {metrics_train['loss']} "
                    f"at the end of iteration {i}"
                )

                # Save checkpoints and best models
                best_dict = retain_best.step(results_valid)
                self._write_weights(
                    {"model": model.state_dict(), "epoch": epoch, "name": model.name},
                    best_dict,
                    fold,
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

            self._erase_tmp(fold)

    def predict(self, fold_list=None):
        pass

    def _find_selection_metrics(self, fold, selection_metric):
        from os import listdir, path

        fold_path = path.join(self.maps_path, f"fold-{fold}")
        if not path.exists(fold_path):
            raise ValueError(
                f"Training of fold {fold} was not performed."
                f"Please execute maps_manager.train(folds={fold})"
            )

        available_metrics = [metric.split("-")[1] for metrics in listdir(fold_path)]
        if selection_metric is None:
            if available_metrics > 1:
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

    def _get_criterion(self):
        from torch import nn

        # TODO: add a check depending on the network task to ensure
        #  the good sizes match of targets / inputs

        loss_dict = {
            "mse": nn.MSELoss(),
            "ce": nn.CrossEntropyLoss(),
        }

        return loss_dict[self.optimization_metric.lower()]

    def _init_model(self, input_shape, resume_fold=None):
        import clinicadl.utils.network as network

        self.logger.info(f"Initialization of model {self.model}")
        # or choose to implement a dictionnary
        model_class = getattr(network, self.model)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        if "input_shape" in args:
            kwargs["input_shape"] = input_shape
            args.remove("input_shape")
        for arg in args:
            kwargs[arg] = getattr(self, arg)

        model = model_class(**kwargs)

        # TODO: implement resume
        # TODO: implement transfer learning

        return model

    def _init_optimizer(self, model, resume_fold=None):
        import torch

        # TODO: implement resume
        optimizer = getattr(torch.optim, self.optimizer)(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer

    ###############################
    # GETTERS                     #
    ###############################
    def get_model(self, fold=0, selection_metric=None):
        """
        Get the model trained corresponding to one fold and one metric evaluated on the validation set.

        Args:
            fold (int): fold number
            selection_metric (str): name of the metric used for the selection
        Returns:
            (Dict): dictionnary of results (weights, epoch number, metrics values)
        """
        from os import path

        import torch

        selection_metric = self._find_selection_metrics(fold, selection_metric)
        model_path = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", "model.pth.tar"
        )
        self.logger.info(
            f"Loading model trained for fold {fold} "
            f"selected according to best validation {selection_metric}."
        )
        return torch.load(model_path)

    def get_prediction(self, prefix, fold=0, selection_metric=None):
        """
        Get the individual predictions for each participant corresponding to one group
        of participants identified by its prefix.

        Args:
            prefix (str): name of the prediction step
            fold (int): fold number
            selection_metric (int): name of the metric used for the selection
        Returns:
            (DataFrame): Results indexed by columns 'participant_id' and 'session_id' which
            identifies the image in the BIDS / CAPS.
        """
        from os import path

        import pandas as pd

        # TODO add prediction log print
        selection_metric = self._find_selection_metrics(fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to prefix {prefix} was found."
            )
        df = pd.read_csv(path.join(prediction_dir, f"{prefix}_results.tsv"), sep="\t")
        df.set_index(["participant_id", "session_id"], inplace=True, drop=True)
        return df

    def get_metrics(self, prefix, fold=0, selection_metric=None):
        """
        Get the metrics corresponding to a group of participants identified by its prefix.

        Args:
            prefix (str): name of the prediction performed on the group of participants
            fold (int): fold number
            selection_metric (int): name of the metric used for the selection
        Returns:
            (Dict[str:float]): Values of the metrics
        """
        from os import path

        import pandas as pd

        # TODO add prediction log print
        selection_metric = self._find_selection_metrics(fold, selection_metric)
        prediction_dir = path.join(
            self.maps_path, f"fold-{fold}", f"best-{selection_metric}", prefix
        )
        if not path.exists(prediction_dir):
            raise ValueError(
                f"No prediction corresponding to prefix {prefix} was found."
            )
        df = pd.read_csv(path.join(prediction_dir, f"{prefix}_metrics.tsv"), sep="\t")
        return df.to_dict("records")[0]
