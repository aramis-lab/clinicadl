import torchio as tio
import json


class Augmentation:
    def __init__(self, name=None, params=None):
        self.name = name
        self.params = params

    def create_augmentation(self):
        augmentation = getattr(tio, self.name)(**self.params)
        return augmentation

    def get_params_dictionary(self):
        return {'name': self.name, 'params': self.params}

    def set_params(self, dict):
        self.name = dict('name')
        self.params = dict('params')


def save_augmentations(augmentation_array, filepath):
    json_dict = {}
    for i, el in enumerate(augmentation_array):
        json_dict[i] = el.get_params_dictionary()

    with open(filepath, 'w') as outfile:
        json.dump(json_dict, outfile)
    print(json_dict)


def load_augmentations(filepath):
    with open(filepath, 'r') as outfile:
        d = json.load(outfile)
    augmentation_array = []

    for key in d.keys():
        augmentation = Augmentation(**d[key])
        augmentation_array.append(augmentation)

    return augmentation_array


def create_tensor_augmentations(augmentations):
    augmentations_tio = []
    for i, el in enumerate(augmentations):
        temp_augm = el.create_augmentation()
        augmentations_tio.append(temp_augm)
    augmentations_tio = tio.OneOf(augmentations_tio)

    return augmentations_tio


##################################
# Transformations
##################################
import torch
import numpy as np


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
            output = torch.nn.functional.pad(image, (-crop[0], crop[0], -crop[1], crop[1]))
        elif dimensions == 3:
            output = torch.nn.functional.pad(image, (-crop[0], crop[0], -crop[1], crop[1], -crop[2], crop[2]))
        else:
            raise ValueError("RandomCropPad is only available for 2D or 3D data.")
        return output


class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        from scipy.ndimage.filters import gaussian_filter

        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample


def get_augmentation_list(data_augmentation):
    augmentation_dict = {
        # "Noise": RandomNoising(sigma=0.1),
        #                  "Erasing": transforms.RandomErasing(),
        #                  "CropPad": RandomCropPad(10),
        #                  "Smoothing": RandomSmoothing(),
        #                  "None": None ,
        'RandomNoise': Augmentation('RandomNoise', {'mean': (-0.03, 0.03), 'std': (0, 0.01)}),
        'RandomBiasField': Augmentation('RandomBiasField', {'coefficients': (0, 0.2), "order": 1}),
        'RandomBiasField2': Augmentation('RandomBiasField', {"coefficients": (0, 0.25), "order": 2}),
        "RandomGamma": Augmentation("RandomGamma", {"log_gamma": (-0.15, 0.15)}),
        "RandomRotation": Augmentation("RandomAffine", {"degrees": (-4, 4), "scales": (1.0, 1.0), "isotropic": True,
                                                        "default_pad_value": 'mean'}),
        "RandomScaling": Augmentation("RandomAffine", {"degrees": (0, 0), "scales": (0.9, 1.1), "isotropic": True,
                                                       "default_pad_value": 'mean'}),
        "RandomRotationAndScaling": Augmentation("RandomAffine",
                                                 {"degrees": (-4, 4), "scales": (0.9, 1.1), "isotropic": True,
                                                  "default_pad_value": 'mean'}),
        "RandomSpike": Augmentation("RandomSpike", {"num_spikes": (1, 2), "intensity": (0.0, 0.08)}),
        "RandomMotion": Augmentation("RandomMotion",
                                     {"degrees": (-0.4, 0.3), "translation": (-0.4, 0.4), "num_transforms": 1}),
        "RandomMotion2": Augmentation("RandomMotion",
                                      {"degrees": (-0.4, 0.3), "translation": (-0.4, 0.4), "num_transforms": 2}),
        "RandomGhosting": Augmentation("RandomGhosting", {"num_ghosts": (2, 4), "intensity": (0.0, 0.2)}),
    }

    augmentation_list = [augmentation_dict[augmentation] for augmentation in data_augmentation]

    return augmentation_list
