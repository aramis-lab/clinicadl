from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import pandas as pd
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.readers import CapsReader
from clinicadl.dictionary.words import GROUPS, PARTICIPANT_ID
from clinicadl.losses.config import LossConfig
from clinicadl.maps.maps import Maps
from clinicadl.metrics import (
    ImplementedMetric,
)
from clinicadl.metrics.config.enum import Optimum
from clinicadl.model import ClinicaDLModel
from clinicadl.networks.config import NetworkConfig
from clinicadl.optim.config import OptimizationConfig
from clinicadl.splitter.split import Split
from clinicadl.splitter.splitter import SingleSplit
from clinicadl.transforms.extraction import Sample
from clinicadl.transforms.extraction.image import ImageSample
from clinicadl.transforms.output_transforms import OutputTransforms
from clinicadl.tsvtools.utils import df_to_tsv, tsv_to_df
from clinicadl.utils.computational.config import ComputationalConfig
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
        # optim_config: Optional[OptimizationConfig] = None,
    ):
        """TO COMPLETE"""

        self.maps = Maps(maps_path)

        self.maps.load()

        if comp_config is None:
            self.comp = ComputationalConfig.from_json(self.maps.computational_json)
        else:
            self.comp = comp_config

        if model is None:
            print(
                "Model is loaded from config class, If you haven't created your model from config class, "
                "please load the model by yourseld and give it as argument to the Predictor"
            )

            self.model = ClinicaDLModel.from_json(self.maps.model_json)
        else:
            self.model = model

    def validate(
        self,
        dataloader: DataLoader[CapsDataset],
        metrics,
        epoch: int = 0,
    ):
        self.model.network.eval()
        dataloader.dataset.eval()  # TODO: check that the dataset is a CapsDataset? or do we accept all kind of dataset ?

        metrics.reset()

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                ############
                images = data.get_images().to(self.comp.device)
                labels = data.get_labels().to(self.comp.device)
                ############

                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    outputs = self.model.network(images)
                    # loss = self.model.loss(outputs, labels)
                    # I think loss is one of callable metrics

                    for callable_metric in metrics.selection_metrics.values():
                        callable_metric(outputs, labels)

            metrics.aggregate(epoch=epoch)

        self.model.network.train()
        return None

    def test(
        self,
        dataloader: DataLoader,
        additionnal_metrics,
        split: int,
        data_group: str,
    ):
        self.model.network.eval()
        df = self.create_prediction_df()
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                if batch == 0:
                    if isinstance(data[0].label, Union[float, int]):
                        df = self.create_prediction_df()
                    elif isinstance(data[0].label, Union[torch.Tensor, None]):
                        caps_reader = self.create_caps_output(
                            split=split, metric=metric, data_group=data_group
                        )

                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    images = data.get_images().to(self.comp.device)
                    outputs = self.model.network(images)

                for i in range(len(data)):
                    if isinstance(data[i].label, Union[float, int]):
                        self.add_sample_pred(df, data[i], outputs[i])

                    elif isinstance(data[i].label, Union[torch.Tensor, None]):
                        self.save_sample_pred(caps_reader, data[i], outputs[i])

        if isinstance(data[0].label, Union[float, int]):
            df.sort_index(inplace=True)
            df.reset_index(inplace=True)
            df.to_csv(
                self.maps.splits[split]
                .best_metrics[metric]
                .data_groups[data_group]
                .prediction_tsv,
                sep="\t",
                index=False,
            )

        self.model.network.train()

        return None

    def add_sample_pred(self, df: pd.DataFrame, sample: Sample, outputs: torch.Tensor):
        df.sort_index(inplace=True)
        df.at[
            (sample.participant, sample.session, sample.id), "ground_truth"
        ] = sample.label

        for i in range(outputs.shape[-1]):
            df.at[
                (sample.participant, sample.session, sample.id), f"proba-{i}"
            ] = outputs[i].item()

        return df

    def create_prediction_df(self):
        df = pd.DataFrame(
            columns=["participant_id", "session_id", "sample_id", "ground_truth"]
        )
        df.set_index(["participant_id", "session_id", "sample_id"], inplace=True)

        return df

    def save_sample_pred(
        self, caps_reader: CapsReader, sample: Sample, output: torch.Tensor
    ):
        sample_path = Path(sample.image_path)

        if not isinstance(sample, ImageSample):
            relative_path = sample_path.relative_to(
                *sample_path.parts[: sample_path.parts.index("subjects") + 1]
            )
            sample_path = Path(
                str(relative_path).replace(
                    f"ses-{sample.session}", f"ses-{sample.session}_sample-{sample.id}"
                )
            )

        (caps_reader.subject_directory / sample_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        output = output.squeeze(0).detach().cpu().float()
        output_nii = nib.Nifti1Image(output.numpy(), affine=sample.affine)
        nib.save(output_nii, (caps_reader.subject_directory / sample_path))

    def create_caps_output(self, split: int, metric: str, data_group: str):
        caps_output_dir = (
            self.maps.splits[split]
            .best_metrics[metric]
            .data_groups[data_group]
            .caps_output
        )

        if caps_output_dir.is_dir():
            raise ValueError(f"Directory {caps_output_dir} already exists")

        caps_output_dir.mkdir(parents=True)
        (caps_output_dir / "subjects").mkdir()
        return CapsReader(caps_output_dir)

    def predict(
        self,
        dataloader: DataLoader[CapsDataset],
        metrics,
        split: int,
        data_group: str,
        transforms: Optional[OutputTransforms] = None,
    ):
        """TO COMPLETE"""

        self._check_leakage(dataset_test=dataloader.dataset)
        # self.create_data_group(metrics, split, data_group)

        self.maps.create_data_group(name=data_group, dataset=dataloader.dataset)

        self.model.network.eval()
        for metric in metrics.val.selection_metrics:
            metric = metric.value
            df = self.create_prediction_df()
            metrics.val.reset()

            with torch.no_grad():
                for batch, data in enumerate(dataloader):
                    if batch == 0:
                        if isinstance(data[0].label, Union[float, int]):
                            df = self.create_prediction_df()
                        elif isinstance(data[0].label, Union[torch.Tensor, None]):
                            caps_reader = self.create_caps_output(
                                split=split, metric=metric, data_group=data_group
                            )

                    # initialize the loss list to save the loss components
                    with autocast(self.comp.device.type, enabled=self.comp.amp):
                        images = data.get_images().to(self.comp.device)
                        labels = data.get_labels().to(self.comp.device)
                        outputs = self.model.network(images)

                    if transforms is not None:
                        outputs = transforms.batch_apply(outputs, data)

                    for callable_metric in metrics.val._callable_metrics.values():
                        callable_metric(outputs, labels)

                    for i in range(len(data)):
                        if isinstance(data[i].label, Union[float, int]):
                            self.add_sample_pred(df, data[i], outputs[i])

                        elif isinstance(data[i].label, Union[torch.Tensor, None]):
                            self.save_sample_pred(caps_reader, data[i], outputs[i])

            metrics.val.aggregate()

            if isinstance(data[0].label, Union[float, int]):
                df.sort_index(inplace=True)
                df.reset_index(inplace=True)
                tsv_path = (
                    self.maps.splits[split]
                    .best_metrics[metric]
                    .data_groups[data_group]
                    .prediction_tsv
                )
                tsv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(tsv_path, sep="\t", index=False)

        self.model.network.train()

        return None

    def _check_leakage(self, dataset_test: CapsDataset):
        """Checks that no intersection exist between the participants used for training and those used for testing."""

        if (
            dataset_test.caps_reader.input_directory.resolve()
            == self.maps.caps_dir().resolve()
        ):  # TODO: add a function to get the caps dir of the czps used for the training from maps reader
            df_train_val = tsv_to_df(self.maps.train_val_tsv)
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
        else:
            print(
                "The inference is done on a different dataset than for training so we are not able to define if there is data leakage or not."
            )
