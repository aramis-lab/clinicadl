import re
from typing import Union

import torchio as tio

from clinicadl.utils.config import ClinicaDLConfig

from ..config import TransformConfig
from ..monai_wrapper import MonaiTransformWrapper
from ..types import Transform, TransformOrConfig

CUSTOM_TRANSFORM = "Custom transform passed by the user"
TO_CONVERT = "to_convert"


class TransformsHandler(ClinicaDLConfig):
    """Base class for Transforms handlers."""

    def _convert_transforms(self):
        """
        Converts configuration classes to actual transforms.

        This operation will be done for all the fields
        """
        for name in self.__private_attributes__:
            pattern = r"^_.*_processed$"
            if re.fullmatch(pattern, name):
                original_field_name = name.removeprefix("_").removesuffix("_processed")
                converted_transforms = self._process_transforms(
                    getattr(self, original_field_name)
                )
                setattr(self, name, converted_transforms)

        return self

    @staticmethod
    def _process_transforms(
        list_transforms: list[TransformOrConfig],
    ) -> Transform:
        """
        Converts TransformConfig objects to transforms, and compose
        transforms.
        """
        only_transforms = []
        for transform in list_transforms:
            if isinstance(transform, TransformConfig):
                real_transform = transform.get_object()
                only_transforms.append(real_transform)
            else:
                only_transforms.append(transform)

        return tio.Compose(only_transforms)

    @classmethod
    def _serialize_transforms(
        cls, transforms: list[TransformOrConfig]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of transforms that are not passed via
        TransformConfigs.
        """
        d = []
        for transform in transforms:
            if isinstance(transform, TransformConfig):
                d.append(transform.to_dict())
            else:
                d.append(
                    CUSTOM_TRANSFORM + ": " + f"'{cls._get_transform_name(transform)}'"
                )

        return d

    @staticmethod
    def _get_transform_name(transform: Transform) -> str:
        """
        Gets a str describing the transform.
        """
        if isinstance(transform, MonaiTransformWrapper):
            transform = transform.transform
        return type(transform).__name__
