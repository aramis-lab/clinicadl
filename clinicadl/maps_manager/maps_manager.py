import json
import subprocess
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast

from clinicadl.caps_dataset.caps_dataset_utils import read_json
from clinicadl.caps_dataset.data import (
    return_dataset,
)
from clinicadl.maps_manager.tmp_config import TmpConfig
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.metrics.utils import (
    check_selection_metric,
    find_selection_metrics,
)
from clinicadl.predict.utils import get_prediction
from clinicadl.trainer.tasks_utils import (
    create_training_config,
    ensemble_prediction,
    evaluation_metrics,
    generate_label_code,
    output_size,
    test,
    test_da,
)
from clinicadl.transforms.config import TransformsConfig
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP, init_ddp
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    MAPSError,
)
from clinicadl.utils.iotools.data_utils import load_data_test
from clinicadl.utils.iotools.maps_manager_utils import (
    add_default_values,
)
from clinicadl.utils.iotools.utils import path_encoder

logger = getLogger("clinicadl.maps_manager")
level_list: List[str] = ["warning", "info", "debug"]


# TODO save weights on CPU for better compatibility


class MapsManager:
    def __init__(
        self,
        maps_path: Path,
        parameters: Optional[Dict[str, Any]] = None,
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

            test_parameters = self.read_maps_json()
            config = TmpConfig(**test_parameters)

            config.n_classes = config.output_size
            if config.network_task == "classification":
                if config.n_classes is None:
                    config.n_classes = output_size(
                        config.network_task, None, config.data_df, config.label
                    )
                config.metrics_module = MetricModule(
                    evaluation_metrics(config.network_task), n_classes=config.n_classes
                )

            elif (
                config.network_task == "regression"
                or config.network_task == "reconstruction"
            ):
                config.metrics_module = MetricModule(
                    evaluation_metrics(config.network_task), n_classes=None
                )

            else:
                raise NotImplementedError(
                    f"Task {config.network_task} is not implemented in ClinicaDL. "
                    f"Please choose between classification, regression and reconstruction."
                )

            config.split_name = (
                self._check_split_wording()
            )  # Used only for retro-compatibility

        # Initiate MAPS
        else:
            config = TmpConfig(**parameters)

            config.split_name = "split"  # Used only for retro-compatibility
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
                self.write_parameters(config.maps_path, self.config.model_dump())
                self._write_requirements_version()
                self._write_training_data()
                self._write_train_val_groups()
                self._write_information()

        self.config = config
        init_ddp(gpu=self.config.gpu, logger=logger)

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.config.model_dump():
            return self.config.model_dump()[name]
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
                    self.config.maps_path
                    / f"{self.config.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                )
                self.write_description_log(
                    log_dir,
                    data_group,
                    dataloader.dataset.config.data.caps_dict,
                    dataloader.dataset.config.data.data_df,
                )

            # load the best trained model during the training
            model, _ = self._init_model(
                transfer_path=self.config.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            model = DDP(
                model, fsdp=self.config.fully_sharded_data_parallel, amp=self.config.amp
            )

            prediction_df, metrics = test(
                mode=self.config.mode,
                metrics_module=self.config.metrics_module,
                n_classes=self.config.n_classes,
                network_task=self.config.network_task,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                use_labels=use_labels,
                amp=amp,
                report_ci=report_ci,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{self.config.mode}_id"] = network

                loss_to_log = (
                    metrics["Metric_values"][-1] if report_ci else metrics["loss"]
                )

                logger.info(
                    f"{self.config.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
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
                self.config.maps_path
                / f"{self.config.split_name}-{split}"
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
                transfer_path=self.config.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            prediction_df, metrics = test_da(
                self.config.network_task,
                model,
                dataloader,
                criterion,
                target=target,
                report_ci=report_ci,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{self.config.mode}_id"] = network

                if report_ci:
                    loss_to_log = metrics["Metric_values"][-1]
                else:
                    loss_to_log = metrics["loss"]

                logger.info(
                    f"{self.config.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
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
            dataset (clinicadl.caps_dataset.data.CapsDataset): wrapper of the data set.
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
                transfer_path=self.config.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=self.config.nb_unfrozen_layer,
            )
            model = DDP(
                model, fsdp=self.config.fully_sharded_data_parallel, amp=self.config.amp
            )
            model.eval()

            tensor_path = (
                self.maps_path
                / f"{self.config.split_name}-{split}"
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
                with autocast("cuda", enabled=self.config.std_amp):
                    output = model(x)
                output = output.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{self.config.mode}_id"]
                input_filename = f"{participant_id}_{session_id}_{self.config.mode}-{mode_id}_input.pt"
                output_filename = f"{participant_id}_{session_id}_{self.config.mode}-{mode_id}_output.pt"
                torch.save(image, tensor_path / input_filename)
                torch.save(output, tensor_path / output_filename)
                logger.debug(f"File saved at {[input_filename, output_filename]}")

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
            selection_metrics = find_selection_metrics(
                self.config.maps_path, self.config.split_name, split
            )

        for selection_metric in selection_metrics:
            #####################
            # Soft voting
            if self.config.num_networks > 1 and not skip_leak_check:
                self._ensemble_to_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
            elif self.config.mode != "image" and not skip_leak_check:
                self._mode_to_image_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )

    ###############################
    # Checks                      #
    ###############################

    def _check_split_wording(self):
        """Finds if MAPS structure uses 'fold-X' or 'split-X' folders."""

        if len(list(self.config.maps_path.glob("fold-*"))) > 0:
            return "fold"
        else:
            return "split"

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
            with (self.config.maps_path / "environment.txt").open(mode="w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def _write_training_data(self):
        """Writes the TSV file containing the participant and session IDs used for training."""
        logger.debug("Writing training data...")

        train_df = load_data_test(
            self.config.tsv_directory,
            self.config.diagnoses,
            baseline=False,
            multi_cohort=self.config.multi_cohort,
        )
        train_df = train_df[["participant_id", "session_id"]]
        if self.config.transfer_path:
            transfer_train_path = (
                self.config.transfer_path / "groups" / "train+validation.tsv"
            )
            transfer_train_df = pd.read_csv(transfer_train_path, sep="\t")
            transfer_train_df = transfer_train_df[["participant_id", "session_id"]]
            train_df = pd.concat([train_df, transfer_train_df])
            train_df.drop_duplicates(inplace=True)
        train_df.to_csv(
            self.config.maps_path / "groups" / "train+validation.tsv",
            sep="\t",
            index=False,
        )

    def _write_train_val_groups(self):
        """Defines the training and validation groups at the initialization"""
        logger.debug("Writing training and validation groups...")
        split_manager = init_split_manager()
        for split in split_manager.split_iterator():
            for data_group in ["train", "validation"]:
                df = split_manager[split][data_group]
                group_path = (
                    self.config.maps_path
                    / "groups"
                    / data_group
                    / f"{self.config.split_name}-{split}"
                )
                group_path.mkdir(parents=True, exist_ok=True)

                columns = ["participant_id", "session_id", "cohort"]
                if self.config.label is not None:
                    columns.append(self.config.label)
                df.to_csv(group_path / "data.tsv", sep="\t", columns=columns)
                self.write_parameters(
                    group_path,
                    {
                        "caps_directory": self.config.caps_directory,
                        "multi_cohort": self.config.multi_cohort,
                    },
                    verbose=False,
                )

    def _write_information(self):
        """
        Writes model architecture of the MAPS in MAPS root.
        """
        from datetime import datetime

        import clinicadl.network as network_package

        model_class = getattr(network_package, self.config.architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = self.config.model_dump()[arg]
        kwargs["gpu"] = False

        model = model_class(**kwargs)

        file_name = "information.log"

        with (self.config.maps_path / file_name).open(mode="w") as f:
            f.write(f"- Date :\t{datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n")
            f.write(f"- Path :\t{self.config.maps_path}\n\n")
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
            self.config.maps_path
            / f"{self.config.split_name}-{split}"
            / f"best-{selection}"
            / data_group
        )
        performance_dir.mkdir(parents=True, exist_ok=True)
        performance_path = (
            performance_dir / f"{data_group}_{self.config.mode}_level_prediction.tsv"
        )
        if not performance_path.is_file():
            results_df.to_csv(performance_path, index=False, sep="\t")
        else:
            results_df.to_csv(
                performance_path, index=False, sep="\t", mode="a", header=False
            )

        metrics_path = (
            performance_dir / f"{data_group}_{self.config.mode}_level_metrics.tsv"
        )
        if metrics is not None:
            # if data_group == "train" or data_group == "validation":
            #     pd_metrics = pd.DataFrame(metrics, index = [0])
            #     header = Trueconfig.
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
        test_df = get_prediction(
            self.config.maps_path,
            self.config.split_name,
            data_group,
            split,
            selection,
            self.config.mode,
            verbose=False,
        )
        validation_df = get_prediction(
            self.config.maps_path,
            self.config.split_name,
            validation_dataset,
            split,
            selection,
            self.config.mode,
            verbose=False,
        )

        performance_dir = (
            self.config.maps_path
            / f"{self.config.split_name}-{split}"
            / f"best-{selection}"
            / data_group
        )

        performance_dir.mkdir(parents=True, exist_ok=True)

        df_final, metrics = ensemble_prediction(
            self.config.mode,
            self.config.metrics_module,
            self.config.n_classes,
            self.config.network_task,
            test_df,
            validation_df,
            selection_threshold=self.config.selection_threshold,
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
        sub_df = get_prediction(
            self.config.maps_path,
            self.config.split_name,
            data_group,
            split,
            selection,
            self.config.mode,
            verbose=False,
        )
        sub_df.rename(columns={f"{self.config.mode}_id": "image_id"}, inplace=True)

        performance_dir = (
            self.config.maps_path
            / f"{self.config.split_name}-{split}"
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
                performance_dir / f"{data_group}_{self.config.mode}_level_metrics.tsv",
                sep="\t",
            )
            if f"{self.config.mode}_id" in metrics_df:
                del metrics_df[f"{self.config.mode}_id"]
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
        transfer_path: Optional[Path] = None,
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
        import clinicadl.network as network_package

        logger.debug(f"Initialization of model {self.config.architecture}")
        # or choose to implement a dictionary
        model_class = getattr(network_package, self.config.architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        for arg in args:
            kwargs[arg] = self.config.model_dump()[arg]

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
                self.config.maps_path
                / f"{self.config.split_name}-{split}"
                / "tmp"
                / "checkpoint.pth.tar"
            )
            checkpoint_state = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
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

    def _init_split_manager_ssda(self, caps_dir, tsv_dir, split_list=None):
        # A intégrer directement dans _init_split_manager
        from clinicadl.validation import split_manager

        split_class = getattr(split_manager, self.config.validation)
        args = list(
            split_class.__init__.__code__.co_varnames[
                : split_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        args.remove("split_list")
        kwargs = {"split_list": split_list}
        for arg in args:
            kwargs[arg] = self.config.model_dump()[arg]

        kwargs["caps_directory"] = Path(caps_dir)
        kwargs["tsv_path"] = Path(tsv_dir)

        return split_class(**kwargs)

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
            self.config.maps_path
            / f"{self.config.split_name}-{split}"
            / f"best-{selection_metric}"
            / data_group
        )
        log_path = log_dir / "description.log"
        with log_path.open(mode="r") as f:
            content = f.read()

    def read_maps_json(self):
        """Returns the training dictionary."""
        json_path = self.config.maps_path / "maps.json"
        return read_json(json_path)

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
        selection_metric = check_selection_metric(
            self.config.maps_path, self.config.split_name, split, selection_metric
        )
        if self.config.multi_network:
            if network is None:
                raise ClinicaDLArgumentError(
                    "Please precise the network number that must be loaded."
                )
            else:
                model_path = (
                    self.config.maps_path
                    / f"{self.config.split_name}-{split}"
                    / f"best-{selection_metric}"
                    / f"network-{network}_model.pth.tar"
                )
        else:
            model_path = (
                self.config.maps_path
                / f"{self.config.split_name}-{split}"
                / f"best-{selection_metric}"
                / "model.pth.tar"
            )

        logger.info(
            f"Loading model trained for split {split} "
            f"selected according to best validation {selection_metric} "
            f"at path {model_path}."
        )
        return torch.load(model_path, map_location=map_location, weights_only=True)

    @property
    def std_amp(self) -> bool:
        """
        Returns whether or not the standard PyTorch AMP should be enabled. It helps
        distinguishing the base DDP with AMP and the usage of FSDP with AMP which
        then calls the internal FSDP AMP mechanisms.
        """
        return self.config.amp and not self.config.fully_sharded_data_parallel
