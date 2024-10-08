from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from clinicadl.maps_manager.maps_manager import MapsManager
from clinicadl.metrics.metric_module import MetricModule
from clinicadl.metrics.utils import find_selection_metrics
from clinicadl.network.network import Network
from clinicadl.trainer.tasks_utils import columns, compute_metrics, generate_test_row
from clinicadl.utils import cluster
from clinicadl.utils.computational.ddp import DDP, init_ddp
from clinicadl.utils.enum import (
    ClassificationLoss,
    ClassificationMetric,
    ReconstructionLoss,
    ReconstructionMetric,
    RegressionLoss,
    RegressionMetric,
    Task,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    MAPSError,
)

logger = getLogger("clinicadl.maps_manager")
level_list: List[str] = ["warning", "info", "debug"]


# TODO save weights on CPU for better compatibility


class Validator:
    def test(
        self,
        mode: str,
        metrics_module: MetricModule,
        n_classes: int,
        network_task,
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        use_labels: bool = True,
        amp: bool = False,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Parameters
        ----------
        model: Network
            The model trained.
        dataloader: DataLoader
            Wrapper of a CapsDataset.
        criterion:  _Loss
            Function to calculate the loss.
        use_labels: bool
            If True the true_label will be written in output DataFrame
            and metrics dict will be created.
        amp: bool
            If True, enables Pytorch's automatic mixed precision.

        Returns
        -------
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()

        results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
        total_loss = {}
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # initialize the loss list to save the loss components
                with autocast("cuda", enabled=amp):
                    outputs, loss_dict = model(data, criterion, use_labels=use_labels)

                if i == 0:
                    for loss_component in loss_dict.keys():
                        total_loss[loss_component] = 0
                for loss_component in total_loss.keys():
                    total_loss[loss_component] += loss_dict[loss_component].float()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = generate_test_row(
                        network_task,
                        mode,
                        metrics_module,
                        n_classes,
                        idx,
                        data,
                        outputs.float(),
                    )
                    row_df = pd.DataFrame(
                        row, columns=columns(network_task, mode, n_classes)
                    )
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
        dataframes = [None] * dist.get_world_size()
        dist.gather_object(
            results_df, dataframes if dist.get_rank() == 0 else None, dst=0
        )
        if dist.get_rank() == 0:
            results_df = pd.concat(dataframes)
        del dataframes
        results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = compute_metrics(
                network_task, results_df, metrics_module, report_ci=report_ci
            )
            for loss_component in total_loss.keys():
                dist.reduce(total_loss[loss_component], dst=0)
                loss_value = total_loss[loss_component].item() / cluster.world_size

                if report_ci:
                    metrics_dict["Metric_names"].append(loss_component)
                    metrics_dict["Metric_values"].append(loss_value)
                    metrics_dict["Lower_CI"].append("N/A")
                    metrics_dict["Upper_CI"].append("N/A")
                    metrics_dict["SE"].append("N/A")

                else:
                    metrics_dict[loss_component] = loss_value

        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def test_da(
        self,
        mode: str,
        metrics_module: MetricModule,
        n_classes: int,
        network_task: Union[str, Task],
        model: Network,
        dataloader: DataLoader,
        criterion: _Loss,
        alpha: float = 0,
        use_labels: bool = True,
        target: bool = True,
        report_ci=False,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Computes the predictions and evaluation metrics.

        Args:
            model: the model trained.
            dataloader: wrapper of a CapsDataset.
            criterion: function to calculate the loss.
            use_labels: If True the true_label will be written in output DataFrame
                and metrics dict will be created.
        Returns:
            the results and metrics on the image level.
        """
        model.eval()
        dataloader.dataset.eval()
        results_df = pd.DataFrame(columns=columns(network_task, mode, n_classes))
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                outputs, loss_dict = model.compute_outputs_and_loss_test(
                    data, criterion, alpha, target
                )
                total_loss += loss_dict["loss"].item()

                # Generate detailed DataFrame
                for idx in range(len(data["participant_id"])):
                    row = generate_test_row(
                        network_task,
                        mode,
                        metrics_module,
                        n_classes,
                        idx,
                        data,
                        outputs,
                    )
                    row_df = pd.DataFrame(
                        row, columns=columns(network_task, mode, n_classes)
                    )
                    results_df = pd.concat([results_df, row_df])

                del outputs, loss_dict
            results_df.reset_index(inplace=True, drop=True)

        if not use_labels:
            metrics_dict = None
        else:
            metrics_dict = compute_metrics(
                network_task, results_df, metrics_module, report_ci=report_ci
            )
            if report_ci:
                metrics_dict["Metric_names"].append("loss")
                metrics_dict["Metric_values"].append(total_loss)
                metrics_dict["Lower_CI"].append("N/A")
                metrics_dict["Upper_CI"].append("N/A")
                metrics_dict["SE"].append("N/A")

            else:
                metrics_dict["loss"] = total_loss

        torch.cuda.empty_cache()

        return results_df, metrics_dict

    def _test_loader(
        self,
        maps_manager: MapsManager,
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
                    maps_manager.maps_path
                    / f"split-{split}"
                    / f"best-{selection_metric}"
                    / data_group
                )
                maps_manager.write_description_log(
                    log_dir,
                    data_group,
                    dataloader.dataset.config.data.caps_dict,
                    dataloader.dataset.config.data.data_df,
                )

            # load the best trained model during the training
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            model = DDP(
                model,
                fsdp=maps_manager.fully_sharded_data_parallel,
                amp=maps_manager.amp,
            )

            prediction_df, metrics = self.test(
                mode=maps_manager.mode,
                metrics_module=maps_manager.metrics_module,
                n_classes=maps_manager.n_classes,
                network_task=maps_manager.network_task,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                use_labels=use_labels,
                amp=amp,
                report_ci=report_ci,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{maps_manager.mode}_id"] = network

                loss_to_log = (
                    metrics["Metric_values"][-1] if report_ci else metrics["loss"]
                )

                logger.info(
                    f"{maps_manager.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            if cluster.master:
                # Replace here
                maps_manager._mode_level_to_tsv(
                    prediction_df,
                    metrics,
                    split,
                    selection_metric,
                    data_group=data_group,
                )

    def _test_loader_ssda(
        self,
        maps_manager: MapsManager,
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
                maps_manager.maps_path
                / f"split-{split}"
                / f"best-{selection_metric}"
                / data_group
            )
            maps_manager.write_description_log(
                log_dir,
                data_group,
                dataloader.dataset.caps_dict,
                dataloader.dataset.df,
            )

            # load the best trained model during the training
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
            )
            prediction_df, metrics = self.test_da(
                network_task=maps_manager.network_task,
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                target=target,
                report_ci=report_ci,
                mode=maps_manager.mode,
                metrics_module=maps_manager.metrics_module,
                n_classes=maps_manager.n_classes,
            )
            if use_labels:
                if network is not None:
                    metrics[f"{maps_manager.mode}_id"] = network

                if report_ci:
                    loss_to_log = metrics["Metric_values"][-1]
                else:
                    loss_to_log = metrics["loss"]

                logger.info(
                    f"{maps_manager.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
                )

            # Replace here
            maps_manager._mode_level_to_tsv(
                prediction_df, metrics, split, selection_metric, data_group=data_group
            )

    @torch.no_grad()
    def _compute_output_tensors(
        self,
        maps_manager: MapsManager,
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
            model, _ = maps_manager._init_model(
                transfer_path=maps_manager.maps_path,
                split=split,
                transfer_selection=selection_metric,
                gpu=gpu,
                network=network,
                nb_unfrozen_layer=maps_manager.nb_unfrozen_layer,
            )
            model = DDP(
                model,
                fsdp=maps_manager.fully_sharded_data_parallel,
                amp=maps_manager.amp,
            )
            model.eval()

            tensor_path = (
                maps_manager.maps_path
                / f"split-{split}"
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
                with autocast("cuda", enabled=maps_manager.std_amp):
                    output = model(x)
                output = output.squeeze(0).cpu().float()
                participant_id = data["participant_id"]
                session_id = data["session_id"]
                mode_id = data[f"{maps_manager.mode}_id"]
                input_filename = f"{participant_id}_{session_id}_{maps_manager.mode}-{mode_id}_input.pt"
                output_filename = f"{participant_id}_{session_id}_{maps_manager.mode}-{mode_id}_output.pt"
                torch.save(image, tensor_path / input_filename)
                torch.save(output, tensor_path / output_filename)
                logger.debug(f"File saved at {[input_filename, output_filename]}")

    def _ensemble_prediction(
        self,
        maps_manager: MapsManager,
        data_group,
        split,
        selection_metrics,
        use_labels=True,
        skip_leak_check=False,
    ):
        """Computes the results on the image-level."""

        if not selection_metrics:
            selection_metrics = find_selection_metrics(maps_manager.maps_path, split)

        for selection_metric in selection_metrics:
            #####################
            # Soft voting
            if maps_manager.num_networks > 1 and not skip_leak_check:
                maps_manager._ensemble_to_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
            elif maps_manager.mode != "image" and not skip_leak_check:
                maps_manager._mode_to_image_tsv(
                    split,
                    selection=selection_metric,
                    data_group=data_group,
                    use_labels=use_labels,
                )
