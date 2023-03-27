"""
Produces a tsv file to study all the nii files and perform the quality check.
"""

from copy import copy

import numpy as np

from clinicadl.utils.caps_dataset.data import MinMaxNormalization


def normalize_homemade(image_np):
    """
    Normalize a numpy image.

    Parameters
    ----------
    image_np: numpy array
        Image to normalize

    Returns
    -------
    image_np: numpy array
        The normalized image.
    """

    for idx, x in np.ndenumerate(image_np):
        image_np[idx] = MinMaxNormalization(image_np[idx])
    return image_np


def distance(contour_np, image_np):
    """
    Return the sum of the pixels in the contour shape for image_np with a threshold of 0.35

    Parameters
    ----------
    contour_np: numpy array
    image_np: numpy array

    Returns
    -------
    sum_in_contour_35: float

    """

    shape3D = image_np.shape
    img_threshold_35 = copy(image_np)
    img_threshold_35[img_threshold_35 < 0.35] = 0

    if not (shape3D == contour_np.shape):
        print("numpy array hasn't the same size and cannot be compare")

    sum_in_contour_35 = 0

    for idx, _ in np.ndenumerate(image_np):
        tmp_threshold_35 = img_threshold_35[idx]
        tmp_contour = int(contour_np[idx])

        if tmp_contour == 1:
            sum_in_contour_35 += tmp_threshold_35

    return sum_in_contour_35
