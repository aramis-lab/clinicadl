"""
Produces a tsv file to study all the nii files and perform the quality check.
"""


from copy import copy

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu


def threshold_otsu_display(image):
    thresh = threshold_otsu(image[np.isfinite(image)])
    binary = image > thresh
    return binary


def normalize_homemade(image_np):
    max_base = image_np.max()
    min_base = image_np.min()
    for idx, x in np.ndenumerate(image_np):
        image_np[idx] = (image_np[idx] - min_base) / (max_base - min_base)
    return image_np


def extract_mask(img_np):

    image_np = normalize_homemade(img_np)

    kernel = np.ones((3, 3), np.uint8)

    mask = threshold_otsu_display(image_np)
    # print(mask.astype("uint8"))
    mask = cv2.dilate(mask.astype("uint8"), kernel, iterations=4)

    mask = cv2.erode(mask.astype("uint8"), kernel, iterations=5)

    mask = binary_fill_holes(mask)

    return mask


def distance(contour_np, brain_np, img_np):

    shape3D = img_np.shape
    img_threshold_35 = copy(img_np)
    img_threshold_35[img_threshold_35 < 0.35] = 0

    mask_np = extract_mask(img_np)

    if not (shape3D == brain_np.shape):
        print("numpy array hasn't the same size and cannot be compare")

    sum_in_brain_35 = 0
    sum_in_contour_35 = 0

    tn_brain = 0
    tp_brain = 0
    fp_brain = 0
    fn_brain = 0

    tn_contour = 0
    tp_contour = 0
    fp_contour = 0
    fn_contour = 0

    for idx, _ in np.ndenumerate(img_np):
        tmp_threshold_35 = img_threshold_35[idx]
        tmp_mask = int(mask_np[idx])
        tmp_brain = int(brain_np[idx])
        tmp_contour = int(contour_np[idx])

        if tmp_brain == 0:

            if tmp_mask == 0:
                tn_brain += 1
            if tmp_mask == 1:
                fp_brain += 1

        elif tmp_brain == 1:
            sum_in_brain_35 += tmp_threshold_35

            if tmp_mask == 1:
                tp_brain += 1
            elif tmp_mask == 0:
                fn_brain += 1

        if tmp_contour == 0:

            if tmp_mask == 0:
                tn_contour += 1
            elif tmp_mask == 1:
                fp_contour += 1

        elif tmp_contour == 1:
            sum_in_contour_35 += tmp_threshold_35

            if tmp_mask == 1:
                tp_contour += 1
            elif tmp_mask == 0:
                fn_contour += 1

    tfp_brain = tp_brain / (fp_brain + tp_brain)
    mttp_contour = 1 - (fp_contour / (fp_contour + tp_contour))
    return tfp_brain, mttp_contour, sum_in_contour_35, sum_in_brain_35
