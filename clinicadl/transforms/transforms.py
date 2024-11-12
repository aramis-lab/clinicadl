from typing import List, Callable

import torchio


class Transforms:
    def __init__(
        self,
        data_augmentation=List[Callable],
        image_transforms=List[Callable],
        object_transforms=List[Callable],
    ) -> None:
        """TO COMPLETE"""
        self.data_augmentation = data_augmentation
