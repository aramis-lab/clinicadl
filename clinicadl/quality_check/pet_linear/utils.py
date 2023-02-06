"""
Produces a tsv file to study all the nii files and perform the quality check.
"""


from copy import copy

import numpy as np


def normalize_homemade(image_np):
    "normalize a numpy image"
    max_base = image_np.max()
    min_base = image_np.min()
    for idx, x in np.ndenumerate(image_np):
        image_np[idx] = (image_np[idx] - min_base) / (max_base - min_base)
    return image_np


def distance(contour_np, img_np):
    "return the sum of the pixels in the contour shape"

    shape3D = img_np.shape
    img_threshold_35 = copy(img_np)
    img_threshold_35[img_threshold_35 < 0.35] = 0

    if not (shape3D == contour_np.shape):
        print("numpy array hasn't the same size and cannot be compare")

    sum_in_contour_35 = 0

    for idx, _ in np.ndenumerate(img_np):
        tmp_threshold_35 = img_threshold_35[idx]
        tmp_contour = int(contour_np[idx])

        if tmp_contour == 1:
            sum_in_contour_35 += tmp_threshold_35

    return sum_in_contour_35
