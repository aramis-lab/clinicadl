from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.dictionary.words import GROUPS, PARTICIPANT_ID
from clinicadl.experiment_manager import ExperimentManager
from clinicadl.experiment_manager.maps_reader import MapsReader
from clinicadl.losses.config import LossConfig
from clinicadl.metrics.metrics import Metrics
from clinicadl.model import ClinicaDLModel
from clinicadl.networks.config import NetworkConfig
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers import OptimizerConfig
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter import SingleSplit
from clinicadl.transforms import OutputTransforms
from clinicadl.tsvtools.utils import tsv_to_df
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
    ClinicaDLDataLeakageError,
)
from clinicadl.utils.typing import PathType


class Predictor:
    def __init__(
        self,
        maps_path: PathType,
        model: Optional[ClinicaDLModel] = None,
        comp_config: Optional[ComputationalConfig] = None,
    ):
        """TO COMPLETE"""

        self.reader = MapsReader(maps_path=maps_path)

        if comp_config is None:
            self.comp = self.reader.get_config(ComputationalConfig)
        else:
            self.comp = comp_config

        if model is None:
            print(
                "Model is loaded from config class, If you haven't created your model from config class, "
                "please load the model by yourseld and give it as argument to the Predictor"
            )
            config_list = self.reader.get_config(
                [NetworkConfig, LossConfig, OptimizerConfig]
            )
            self.model = ClinicaDLModel.from_config(
                network_config=config_list[0],
                loss_config=config_list[1],
                optimizer_config=config_list[2],
            )
        else:
            self.model = model

    def test(
        self,
        dataloader: DataLoader,
        metrics: Metrics,
        transforms: Optional[OutputTransforms] = None,
        epoch: int = 0,
    ):
        self.model.network.eval()
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                ############
                images = data.get_images().to(self.comp.device)
                labels = data.get_labels().to(self.comp.device)
                ############

                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = self.model.network(images)
                    loss = self.model.loss(outputs, labels)

                if transforms:
                    if len(outputs.shape) != len(images.shape):
                        raise ValueError(
                            "Outputs tensors can only be applied if the outputs is of same size as the input"
                        )

                    else:
                        (outputs, labels) = transforms.batch_apply(outputs, data)
                        # TODO: check if apply on label but I think it is applied on all the sample/dataPoint -> it depends on the transforms

                metrics.val.compute(
                    batch=batch, epoch=epoch, data=(outputs, labels), loss=loss
                )
                print(loss)
                print(metrics.val.get_loss(batch=batch, epoch=epoch))
        self.model.network.train()
        return None

    def predict(
        self,
        dataset: CapsDataset,
        split_dir: Path,
        data_loader_config: DataLoaderConfig,
        metrics: Metrics,
        transforms: Optional[OutputTransforms] = None,
    ):
        """TO COMPLETE"""

        splitter = SingleSplit(split_dir=split_dir)
        split = splitter.get_splits(dataset=dataset)

        self._check_leakage(dataset_test=split.val_dataset)
        if data_loader_config is None:
            data_loader_config = self.reader.get_data_loader_config()
        split.build_val_loader(dataloader_config=data_loader_config)
        self.test(split.val_loader, metrics, transforms)

        return None

    def _check_leakage(self, dataset_test: CapsDataset):
        """Checks that no intersection exist between the participants used for training and those used for testing."""

        df_train_val = self.reader.get_train_val_df()
        df_test = dataset_test.df

        participants_train = set(df_train_val[PARTICIPANT_ID].values)
        participants_test = set(df_test[PARTICIPANT_ID].values)
        intersection = participants_test & participants_train

        if len(intersection) > 0:
            raise ClinicaDLDataLeakageError(
                "Your evaluation set contains participants who were already seen during "
                "the training step. The list of common participants is the following: "
                f"{intersection}."
            )

    # def _test_loader(
    #     self,
    #     maps_manager: MapsManager,
    #     dataloader,
    #     criterion,
    #     data_group: str,
    #     split: int,
    #     selection_metrics,
    #     use_labels=True,
    #     gpu=None,
    #     amp=False,
    #     network=None,
    #     report_ci=True,
    # ):
    #     """
    #     Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files.

    #     Args:
    #         dataloader (torch.utils.data.DataLoader): DataLoader wrapping the test CapsDataset.
    #         criterion (torch.nn.modules.loss._Loss): optimization criterion used during training.
    #         data_group (str): name of the data group used for the testing task.
    #         split (int): Index of the split used to train the model tested.
    #         selection_metrics (list[str]): List of metrics used to select the best models which are tested.
    #         use_labels (bool): If True, the labels must exist in test meta-data and metrics are computed.
    #         gpu (bool): If given, a new value for the device of the model will be computed.
    #         amp (bool): If enabled, uses Automatic Mixed Precision (requires GPU usage).
    #         network (int): Index of the network tested (only used in multi-network setting).
    #     """
    #     for selection_metric in selection_metrics:
    #         if cluster.master:
    #             log_dir = (
    #                 maps_manager.maps_path
    #                 / f"split-{split}"
    #                 / f"best-{selection_metric}"
    #                 / data_group
    #             )
    #             maps_manager.write_description_log(
    #                 log_dir,
    #                 data_group,
    #                 dataloader.dataset.config.data.caps_dict,
    #                 dataloader.dataset.config.data.data_df,
    #             )

    #         # load the best trained model during the training
    #         model, _ = maps_manager._init_model(
    #             transfer_path=maps_manager.maps_path,
    #             split=split,
    #             transfer_selection=selection_metric,
    #             gpu=gpu,
    #             network=network,
    #         )
    #         model = DDP(
    #             model,
    #             fsdp=maps_manager.fully_sharded_data_parallel,
    #             amp=maps_manager.amp,
    #         )

    #         prediction_df, metrics = self.test(
    #             mode=maps_manager.mode,
    #             metrics_module=maps_manager.metrics_module,
    #             n_classes=maps_manager.n_classes,
    #             network_task=maps_manager.network_task,
    #             model=model,
    #             dataloader=dataloader,
    #             criterion=criterion,
    #             use_labels=use_labels,
    #             amp=amp,
    #             report_ci=report_ci,
    #         )
    #         if use_labels:
    #             if network is not None:
    #                 metrics[f"{maps_manager.mode}_id"] = network

    #             loss_to_log = (
    #                 metrics["Metric_values"][-1] if report_ci else metrics["loss"]
    #             )

    #             logger.info(
    #                 f"{maps_manager.mode} level {data_group} loss is {loss_to_log} for model selected on {selection_metric}"
    #             )

    #         if cluster.master:
    #             # Replace here
    #             maps_manager._mode_level_to_tsv(
    #                 prediction_df,
    #                 metrics,
    #                 split,
    #                 selection_metric,
    #                 data_group=data_group,
    #             )

    # def _test_loader(self):
    #     """Launches the testing task on a dataset wrapped by a DataLoader and writes prediction TSV files."""
    #     pass

    # def _compute_latent_tensor(self):
    #     """Compute the output tensors and saves them in the MAPS."""
    #     pass

    # @torch.no_grad()
    # def _compute_output_nifti(self):
    #     """omputes the output nifti images and saves them in the MAPS."""
    #     pass

    # @torch.no_grad()
    # def _compute_output_tensors(self):
    #     """Compute the output tensors and saves them in the MAPS."""
    #     pass

    # def _ensemble_prediction(self):
    #     """Computes the results on the image-level."""
    #     pass

    # def _get_prediction(
    #     self,
    #     data_group: str,
    #     split: int = 0,
    #     selection_metric: str = "loss",
    #     mode: str = "image",  # TODO : need to change this to an ExtractionConfig
    #     verbose: bool = False,  # TODO: do we remove verbose argument everywhere ?
    # ):
    #     """
    #     Get the individual predictions for each participant corresponding to one group
    #     of participants identified by its data group.

    #     Args:
    #         data_group (str): name of the data group used for the prediction task.
    #         split (int): Index of the split used for training.
    #         selection_metric (str): Metric used for best weights selection.
    #         mode (str): level of the prediction.
    #         verbose (bool): if True will print associated prediction.log.
    #     Returns:
    #         (DataFrame): Results indexed by columns 'participant_id' and 'session_id' which
    #         identifies the image in the BIDS / CAPS.
    #     """
    #     selection_metric = check_selection_metric(
    #         self.maps_path, split, selection_metric
    #     )
    #     if verbose:
    #         self.print_description_log(split, selection_metric, data_group)

    #     if not self.data_group_dir(
    #         split=split, selection_metric=selection_metric, data_group=data_group
    #     ).is_dir():
    #         raise MAPSError(
    #             f"No prediction corresponding to data group {data_group} was found."
    #         )
    #     df = pd.read_csv(
    #         self.prediction_tsv(
    #             split=split,
    #             selection_metric=selection_metric,
    #             data_group=data_group,
    #             mode=mode,
    #         ),
    #         sep="\t",
    #     )
    #     df.set_index(self.df_index, inplace=True, drop=True)
    #     return df
