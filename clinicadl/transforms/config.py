from logging import getLogger
from typing import Optional, Tuple

import torchvision.transforms as torch_transforms
from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.transforms import transforms
from clinicadl.utils.enum import (
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.training_config")


class TransformsConfig(BaseModel):  # TODO : put in data module?
    """Config class to handle the transformations applied to th data."""

    data_augmentation: Tuple[Transform, ...] = ()
    train_transformations: Optional[Tuple[Transform, ...]] = None
    normalize: bool = True
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @field_validator("data_augmentation", mode="before")
    def validator_data_augmentation(cls, v):
        """Transforms lists to tuples and False to empty tuple."""
        if isinstance(v, list):
            return tuple(v)
        if v is False:
            return ()
        return v

    def get_transforms(
        self,
    ) -> Tuple[torch_transforms.Compose, torch_transforms.Compose]:
        """
        Outputs the transformations that will be applied to the dataset

        Args:
            normalize: if True will perform MinMaxNormalization.
            data_augmentation: list of data augmentation performed on the training set.

        Returns:
            transforms to apply in train and evaluation mode / transforms to apply in evaluation mode only.
        """
        augmentation_dict = {
            "Noise": transforms.RandomNoising(sigma=0.1),
            "Erasing": torch_transforms.RandomErasing(),
            "CropPad": transforms.RandomCropPad(10),
            "Smoothing": transforms.RandomSmoothing(),
            "Motion": transforms.RandomMotion((2, 4), (2, 4), 2),
            "Ghosting": transforms.RandomGhosting((4, 10)),
            "Spike": transforms.RandomSpike(1, (1, 3)),
            "BiasField": transforms.RandomBiasField(0.5),
            "RandomBlur": transforms.RandomBlur((0, 2)),
            "RandomSwap": transforms.RandomSwap(15, 100),
            "None": None,
        }

        augmentation_list = []
        transformations_list = []

        if self.data_augmentation:
            augmentation_list.extend(
                [
                    augmentation_dict[augmentation]
                    for augmentation in self.data_augmentation
                ]
            )

        transformations_list.append(transforms.NanRemoval())
        if self.normalize:
            transformations_list.append(transforms.MinMaxNormalization())
        if self.size_reduction:
            transformations_list.append(
                transforms.SizeReduction(self.size_reduction_factor)
            )

        all_transformations = torch_transforms.Compose(transformations_list)
        train_transformations = torch_transforms.Compose(augmentation_list)

        return train_transformations, all_transformations
