"""Config classes for transforms supported natively in ``ClinicaDL``, for preprocessing and augmentation. Based
on :torchio:`TorchIO transforms <transforms/transforms.html>`."""

from .base import OneOfConfig, TransformConfig
from .enum import ImplementedTransform
from .factory import get_transform_config
from .intensity import *
from .intensity_augmentations import *
from .label import *
from .post_processing import *
from .spatial import *
from .spatial_augmentations import *
