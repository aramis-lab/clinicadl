from __future__ import annotations

import random
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.callbacks import Callback
from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint

if TYPE_CHECKING:
    from clinicadl.data.dataloader import BatchType


class ResampleMask(tio.SpatialTransform):
    """
    To fake a resampling.
    """

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint["leftHemisphere"].affine = datapoint.image.affine

        return datapoint


class RandomMasking(tio.IntensityTransform):
    """
    Randomly masks one half of the image, along axial
    or coronal axis.
    """

    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        image: tio.ScalarImage = datapoint.image
        label = np.ones(6)  # (L, R, P, A, I, S)

        apply_common_mask = random.random() >= 0.5
        if apply_common_mask:
            label[0] = 0
            common_mask: tio.LabelMap = datapoint["leftHemisphere"]
            image.data *= common_mask.data

        apply_individual_mask = random.random() >= 0.5
        if apply_individual_mask:
            mask: tio.LabelMap = datapoint["head"]
            direction = random.randint(1, 2)
            before_middle = random.random() >= 0.5
            label[direction * 2 + int(not before_middle)] = 0

            idx = [slice(None)] * mask.data.ndim
            middle = mask.spatial_shape[direction] // 2
            idx[direction + 1] = (
                slice(None, middle) if before_middle else slice(middle, None)
            )
            mask.data[idx] = 0

            image.data *= mask.data

        datapoint["label"] = label

        return datapoint


class TestModelReset(Callback):
    def __init__(self, assert_equal: bool):
        self.nn_state_dict = None
        self.assert_equal = assert_equal

    def on_train_start(self, *, model: torch.nn.Module, **kwargs):
        if self.nn_state_dict is None:
            self.nn_state_dict = deepcopy(model.state_dict())
        else:
            if self.assert_equal:
                torch.testing.assert_close(self.nn_state_dict, model.state_dict())
            else:
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(self.nn_state_dict, model.state_dict())


class ErrorCallback(Callback):
    def __init__(self, error_epoch: int):
        self.error_epoch = error_epoch
        self.error_raised = False

    def on_backward_step_start(self, *, state, **kwargs):
        if state.current_epoch == self.error_epoch and not self.error_raised:
            self.error_raised = True
            raise torch.cuda.OutOfMemoryError()


class TestDeviceCallback(Callback):
    def __init__(
        self,
        model_on_gpu: Optional[bool] = None,
        post_processing_on_gpu: Optional[bool] = None,
        metrics_on_gpu: Optional[bool] = None,
    ):
        self.model_on_gpu = model_on_gpu
        self.post_processing_on_gpu = post_processing_on_gpu
        self.metrics_on_gpu = metrics_on_gpu
        self.batch = None

    def on_evaluation_step_start(self, *, batch, **kwargs):
        self._check(batch, self.model_on_gpu)

    def on_metrics_computation_start(self, *, output, **kwargs):
        self.batch = output
        self._check(output, self.post_processing_on_gpu)

    def on_metrics_computation_end(self, **kwargs):
        self._check(self.batch, self.metrics_on_gpu)

    @staticmethod
    def _check(batch: BatchType, gpu: Optional[bool]) -> None:
        if not isinstance(batch, Batch):
            batch = batch[0]
        if gpu is not None:
            assert batch.device == (
                torch.device("cuda") if gpu else torch.device("cpu")
            )
