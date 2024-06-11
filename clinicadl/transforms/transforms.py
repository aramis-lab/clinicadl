# coding: utf8

from logging import getLogger
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchio as tio
import torchvision.transforms as transforms

from clinicadl.utils.exceptions import (
    ClinicaDLConfigurationError,
)

logger = getLogger("clinicadl")

##################################
# Transformations
##################################


class RandomNoising(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, image):
        import random

        sigma = random.uniform(0, self.sigma)
        dist = torch.distributions.normal.Normal(0, sigma)
        return image + dist.sample(image.shape)


class RandomSmoothing(object):
    """Applies a random zoom to a tensor"""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, image):
        import random

        from scipy.ndimage import gaussian_filter

        sigma = random.uniform(0, self.sigma)
        image = gaussian_filter(image, sigma)  # smoothing of data
        image = torch.from_numpy(image).float()
        return image


class RandomCropPad(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, image):
        dimensions = len(image.shape) - 1
        crop = np.random.randint(-self.length, self.length, dimensions)
        if dimensions == 2:
            output = torch.nn.functional.pad(
                image, (-crop[0], crop[0], -crop[1], crop[1])
            )
        elif dimensions == 3:
            output = torch.nn.functional.pad(
                image, (-crop[0], crop[0], -crop[1], crop[1], -crop[2], crop[2])
            )
        else:
            raise ValueError(
                f"RandomCropPad is only available for 2D or 3D data. Image is {dimensions}D"
            )
        return output


class GaussianSmoothing(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        from scipy.ndimage.filters import gaussian_filter

        image = sample["image"]
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample["image"] = smoothed_image

        return sample


class RandomMotion(object):
    """Applies a Random Motion"""

    def __init__(self, translation, rotation, num_transforms):
        self.rotation = rotation
        self.translation = translation
        self.num_transforms = num_transforms

    def __call__(self, image):
        motion = tio.RandomMotion(
            degrees=self.rotation,
            translation=self.translation,
            num_transforms=self.num_transforms,
        )
        image = motion(image)

        return image


class RandomGhosting(object):
    """Applies a Random Ghosting"""

    def __init__(self, num_ghosts):
        self.num_ghosts = num_ghosts

    def __call__(self, image):
        ghost = tio.RandomGhosting(num_ghosts=self.num_ghosts)
        image = ghost(image)

        return image


class RandomSpike(object):
    """Applies a Random Spike"""

    def __init__(self, num_spikes, intensity):
        self.num_spikes = num_spikes
        self.intensity = intensity

    def __call__(self, image):
        spike = tio.RandomSpike(
            num_spikes=self.num_spikes,
            intensity=self.intensity,
        )
        image = spike(image)

        return image


class RandomBiasField(object):
    """Applies a Random Bias Field"""

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, image):
        bias_field = tio.RandomBiasField(coefficients=self.coefficients)
        image = bias_field(image)

        return image


class RandomBlur(object):
    """Applies a Random Blur"""

    def __init__(self, std):
        self.std = std

    def __call__(self, image):
        blur = tio.RandomBlur(std=self.std)
        image = blur(image)

        return image


class RandomSwap(object):
    """Applies a Random Swap"""

    def __init__(self, patch_size, num_iterations):
        self.patch_size = patch_size
        self.num_iterations = num_iterations

    def __call__(self, image):
        swap = tio.RandomSwap(
            patch_size=self.patch_size, num_iterations=self.num_iterations
        )
        image = swap(image)

        return image


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())


class NanRemoval(object):
    def __init__(self):
        self.nan_detected = False  # Avoid warning each time new data is seen

    def __call__(self, image):
        if torch.isnan(image).any().item():
            if not self.nan_detected:
                logger.warning(
                    "NaN values were found in your images and will be removed."
                )
                self.nan_detected = True
            return torch.nan_to_num(image)
        else:
            return image


class SizeReduction(object):
    """Reshape the input tensor to be of size [80, 96, 80]"""

    def __init__(self, size_reduction_factor=2) -> None:
        self.size_reduction_factor = size_reduction_factor

    def __call__(self, image):
        if self.size_reduction_factor == 2:
            return image[:, 4:164:2, 8:200:2, 8:168:2]
        elif self.size_reduction_factor == 3:
            return image[:, 0:168:3, 8:200:3, 4:172:3]
        elif self.size_reduction_factor == 4:
            return image[:, 4:164:4, 8:200:4, 8:168:4]
        elif self.size_reduction_factor == 5:
            return image[:, 4:164:5, 0:200:5, 8:168:5]
        else:
            raise ClinicaDLConfigurationError(
                "size_reduction_factor must be 2, 3, 4 or 5."
            )
