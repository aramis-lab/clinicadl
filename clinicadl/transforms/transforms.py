from typing import List, Optional

import torchio.transforms as transforms
from torchio.transforms.transform import Transform

from clinicadl.dataset.config.extraction import ExtractionConfig
from clinicadl.utils.enum import ExtractionMethod


class Transforms:
    def __init__(
        self,
        data_augmentation: Optional[list[Transform]] = None,
        image_transforms: Optional[list[Transform]] = None,
        object_transforms: Optional[list[Transform]] = None,
        extraction_method: Optional[ExtractionMethod] = ExtractionMethod.IMAGE,
    ) -> None:
        """TO COMPLETE"""

        if data_augmentation:
            self.data_augmentation = data_augmentation

        if image_transforms:
            self.image_transforms = image_transforms

        if object_transforms:
            self.object_transforms = object_transforms

        if extraction_method not in ExtractionMethod:
            raise ValueError(f"Invalid extraction method: {extraction_method}")
