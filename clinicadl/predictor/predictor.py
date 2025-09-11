# TODO : clean and improve the predictor but it gives a good idea to start

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.metrics.metric import CumulativeIterationMetric as MonaiMetric
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.readers import CapsReader
from clinicadl.dictionary.words import LOSS, PARTICIPANT_ID
from clinicadl.IO.maps.maps import Maps
from clinicadl.losses.config import LossConfig
from clinicadl.losses.types import Loss
from clinicadl.metrics.config import MetricConfig
from clinicadl.metrics.handler import LossMetricConfig
from clinicadl.models import ClinicaDLModel
from clinicadl.transforms.extraction import Sample
from clinicadl.transforms.extraction.image import ImageSample
from clinicadl.transforms.handlers import Postprocessing, Transforms
from clinicadl.tsvtools.utils import tsv_to_df
from clinicadl.utils.computational.config import ComputationalConfig
from clinicadl.utils.exceptions import ClinicaDLDataLeakageError
from clinicadl.utils.typing import PathType


class Predictor:
    """
    TO COMPLETE
    """

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
            self.comp = ComputationalConfig.from_json(
                self.maps.training.computational_json
            )
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
        split: int,
        data_group: str,
        additionnal_metrics: Optional[
            list[Union[MetricConfig, MonaiMetric, LossMetricConfig, LossConfig, Loss]]
        ] = None,
        output_transforms: Optional[OutputTransforms] = None,
        metric: str = "LossMetric",
    ):
        # TODO : check dataloader with dataloader from training

        # TODO : check the model weights from the split given exists

        # TODO : check the datagroup doesn't exist for these parameters

        # TODO : retrieve the metrics from the maps and add the additional metrics
        # metrics = MetricsHandler.from_maps(additionnal_metrics= additionnal_metrics)

        # TODO : check that the Transforms is of Type OutputsTransforms, if not put the transforms in an OutputsTransforms Object

        self.maps.predictions._create_group(
            group_name=data_group,
            split=split,
            dataset=dataloader.dataset,
            metric=metric,
        )

        # TODO : create a new method to create a new datagroup

        self.model.network.eval()
        is_caps_output = False

        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                if batch_idx == 0:
                    if isinstance(data[0].label, (float, int, np.floating, np.integer)):
                        prediction_df = self.create_prediction_df()
                    elif isinstance(data[0].label, torch.Tensor) or (
                        data[0].label is None
                    ):
                        is_caps_output = True
                        caps_reader = self.create_caps_output(
                            split=split, metric=metric, data_group=data_group
                        )
                    else:
                        raise NotImplementedError(
                            f"Output type {type(data[0].label)} is not supported yet."
                        )

                # initialize the loss list to save the loss components
                with autocast(self.comp.device.type, enabled=self.comp.amp):
                    images = data.get_images().to(self.comp.device)
                    outputs = self.model.network(images)

                # if output_transforms is not None: # TODO: apply the output transforms
                #     outputs = output_transforms.batch_apply(outputs, data)

                # metrics=(outputs, labels) # TODO: check what we wanrt to pass to the metrics

                for i, sample in enumerate(data):
                    if is_caps_output:
                        self.save_sample_pred(caps_reader, sample, outputs[i])

                    else:
                        self.add_sample_pred(prediction_df, sample, outputs[i])

        if not is_caps_output:
            prediction_df.sort_index(inplace=True)
            prediction_df.reset_index(inplace=True)
            prediction_df.to_csv(
                self.maps.splits[split]
                .best_metrics[metric]
                .data_groups[data_group]
                .predictions_tsv,
                sep="\t",
                index=False,
            )

        self.model.network.train()

    def add_sample_pred(self, df: pd.DataFrame, sample: Sample, outputs: torch.Tensor):
        """TO COMPLETE"""

        df.sort_index(inplace=True)
        df.at[
            (sample.participant, sample.session, sample._sample_index), "ground_truth"
        ] = sample.label

        for i in range(outputs.shape[-1]):
            df.at[
                (sample.participant, sample.session, sample._sample_index), f"proba-{i}"
            ] = outputs[i].item()

        return df

    def create_prediction_df(self):
        """TO COMPLETE"""
        df = pd.DataFrame(
            columns=["participant_id", "session_id", "sample_id", "ground_truth"]
        )
        df.set_index(["participant_id", "session_id", "sample_id"], inplace=True)

        return df

    def save_sample_pred(
        self, caps_reader: CapsReader, sample: Sample, output: torch.Tensor
    ):
        """TO COMPLETE"""
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
        output_nii = nib.nifti1.Nifti1Image(output.numpy(), affine=sample.affine)
        nib.loadsave.save(output_nii, (caps_reader.subject_directory / sample_path))

    def create_caps_output(self, split: int, metric: str, data_group: str):
        """TO COMPLETE"""
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
