from typing import List

import torchio


class Transforms:
    def __init__(
        self,
        data_augmentation=List[torchio.Transform],
        image_transforms=List[torchio.Transform],
        object_transforms=List[torchio.Transform],
    ) -> None:
        """TO COMPLETE"""
        self.data_augmentation = data_augmentation
