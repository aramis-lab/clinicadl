"""
Produces a tsv file to study all the nii files and perform the quality check.
"""

from copy import copy

import numpy as np

from clinicadl.utils.caps_dataset.data import MinMaxNormalization


def get_metric(contour_np, image_np, inside):
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
    img_copy = copy(image_np)
    img_norm_thresh = MinMaxNormalization()(img_copy)
    img_norm_thresh[img_norm_thresh < 0.35] = 0

    if not (shape3D == contour_np.shape):
        raise ValueError(f"The images must be of shape: {shape3D}")

    sum_in_contour = 0

    for idx, _ in np.ndenumerate(image_np):
        tmp_norm_tresh = img_norm_thresh[idx]
        tmp_contour = int(contour_np[idx])

        if tmp_contour == 1:
            sum_in_contour += tmp_norm_tresh

    return 1 - (sum_in_contour / inside)
