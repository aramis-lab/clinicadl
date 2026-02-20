import random
from pathlib import Path

import numpy as np
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import T1Linear
from clinicadl.data.structures import DataPoint
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    OneOfConfig,
    RandomAffineConfig,
    RandomBiasFieldConfig,
    RandomBlurConfig,
    RandomElasticDeformationConfig,
    RandomGammaConfig,
    RandomGhostingConfig,
    RandomMotionConfig,
    RandomNoiseConfig,
    RandomSpikeConfig,
)


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


def build_dataset(dir_: Path) -> CapsDataset:
    dataset = CapsDataset(
        directory=dir_,
        datatype=T1Linear(use_uncropped_image=False),
        data=dir_ / "metadata.tsv",
        masks=["leftHemisphere.nii.gz", "head"],
        columns=["age"],
        transforms=TransformsHandler(
            sample_transforms=[ResampleMask(), RandomMasking()],
            augmentations=[
                OneOfConfig(
                    transforms=[
                        RandomAffineConfig(),
                        RandomElasticDeformationConfig(),
                        RandomMotionConfig(),
                        RandomGhostingConfig(),
                        RandomGammaConfig(),
                        RandomSpikeConfig(),
                        RandomBiasFieldConfig(),
                        RandomBlurConfig(),
                        RandomNoiseConfig(),
                    ]
                ),
            ],
        ),
    )
    dataset.read_tensor_conversion()

    return dataset
