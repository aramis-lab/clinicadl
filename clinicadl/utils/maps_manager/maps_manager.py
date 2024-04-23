import json
import shutil
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from clinicadl.utils.caps_dataset.data import (
    get_transforms,
    return_dataset,
)
from clinicadl.utils.cmdline_utils import check_gpu
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    MAPSError,
)
from clinicadl.utils.maps_manager.ddp import DDP, cluster, init_ddp
from clinicadl.utils.maps_manager.maps_manager_utils import (
    add_default_values,
    read_json,
)
from clinicadl.utils.metric_module import RetainBest
from clinicadl.utils.preprocessing import path_encoder
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
            # test_parameters = path_decoder(test_parameters)
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

    ###################################
    # High-level functions templates  #
    ###################################
    def _test_loader(
        self,
        dataloader,
        criterion,
        data_group: str,
        split: int,
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
            model = DDP(model, fsdp=self.fully_sharded_data_parallel, amp=self.amp)

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
            model = DDP(model, fsdp=self.fully_sharded_data_parallel, amp=self.amp)
            model.eval()

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

            for i in [
                *range(cluster.rank, nb_modes, cluster.world_size),
                *range(int(nb_modes % cluster.world_size <= cluster.rank)),
            ]:
                data = dataset[i]
                image = data["image"]
                x = image.unsqueeze(0).to(model.device)
                with autocast(enabled=self.std_amp):
                    output = model(x)
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

    def _find_splits(self):
        """Find which splits were trained in the MAPS."""
        return [
            int(split.name.split("-")[1])
            for split in list(self.maps_path.iterdir())
            if split.name.startswith(f"{self.split_name}-")
        ]

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
        print("################################")
        print(available_metrics)
        print(split)
        print("################################")
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

    ###############################
    # File writers                #
    ###############################
    @staticmethod
    def write_parameters(json_path: Path, parameters, verbose=True):
        """Write JSON files of parameters."""
        logger.debug("Writing parameters...")
        json_path.mkdir(parents=True, exist_ok=True)

        # save to json file
        json_path = json_path / "maps.json"
        if verbose:
            logger.info(f"Path of json file: {json_path}")

        with json_path.open(mode="w") as json_file:
            json.dump(
                parameters, json_file, skipkeys=True, indent=4, default=path_encoder
            )

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

    def _init_split_manager(self, split_list=None, ssda_bool: bool = False):
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

        if ssda_bool:
            kwargs["caps_directory"] = self.caps_target
            kwargs["tsv_path"] = self.tsv_target_lab

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

    def _init_task_manager(
        self, df: Optional[pd.DataFrame] = None, n_classes: Optional[int] = None
    ):
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

    ###############################
    # Getters                     #
    ###############################
    def _print_description_log(
        self,
        data_group: str,
        split: int,
        selection_metric: str,
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

    def get_parameters(self):
        """Returns the training parameters dictionary."""
        json_path = self.maps_path / "maps.json"
        return read_json(json_path)

    # never used ??
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
        self,
        split=0,
        selection_metric: Optional[str] = None,
        network: Optional[int] = None,
        map_location: Optional[str] = None,
    ):
        """
        Get the model trained corresponding to one split and one metric evaluated on the validation set.

        Parameters
        ----------
        split: int
            Index of the split used for training.
        selection_metric: str
            name of the metric used for the selection.
        network: int
            Index of the network trained (used in multi-network setting only).
        map_location: str
            torch.device object or a string containing a device tag,
            it indicates the location where all tensors should be loaded.
            (see https://pytorch.org/docs/stable/generated/torch.load.html).

        Returns
        -------
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
        self,
        data_group: str,
        split: int = 0,
        selection_metric: Optional[str] = None,
        mode: str = "image",
        verbose: bool = False,
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
        self,
        data_group: str,
        split: int = 0,
        selection_metric: Optional[str] = None,
        mode: str = "image",
        verbose: bool = True,
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

    @property
    def std_amp(self) -> bool:
        """
        Returns whether or not the standard PyTorch AMP should be enabled. It helps
        distinguishing the base DDP with AMP and the usage of FSDP with AMP which
        then calls the internal FSDP AMP mechanisms.
        """
        return self.amp and not self.fully_sharded_data_parallel
