from typing import List

import torchio


class Transforms:
    def __init__(
        self,
        data_augmentation=List[torchio],
        image_transforms=List[torchio],
        object_transforms=List[torchio],
    ) -> None:
        """TO COMPLETE"""
        self.data_augmentation = data_augmentation
