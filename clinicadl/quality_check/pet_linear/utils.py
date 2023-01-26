"""
Produces a tsv file to study all the nii files and perform the quality check.
"""


from copy import copy

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu


def thresholdOtsuDisplay(image):
    thresh = threshold_otsu(image[np.isfinite(image)])
    binary = image > thresh
    return binary


def normalize_homemade(image_np):
    max_base = image_np.max()
    min_base = image_np.min()
    for i in range(len(image_np)):
        for j in range(len(image_np[i])):
            for k in range(len(image_np[i][j])):

                image_np[i][j][k] = (image_np[i][j][k] - min_base) / (
                    max_base - min_base
                )
    return image_np


def extract_mask(img_np):

    image_np = normalize_homemade(img_np)

    kernel = np.ones((3, 3), np.uint8)

    mask = thresholdOtsuDisplay(image_np)
    # print(mask.astype("uint8"))
    mask = cv2.dilate(mask.astype("uint8"), kernel, iterations=4)

    mask = cv2.erode(mask.astype("uint8"), kernel, iterations=5)

    mask = binary_fill_holes(mask)

    return mask


def distance(contour_np, brain_np, img_np):

    shape3D = img_np.shape
    img_seuil_35 = copy(img_np)
    img_seuil_35[img_seuil_35 < 0.35] = 0

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

    xx = int(shape3D[0])
    yy = shape3D[1]
    zz = shape3D[2]

    for i in range(xx):
        for j in range(yy):
            for k in range(zz):

                tmp_seuil_35 = img_seuil_35[i][j][k]
                tmp_mask = int(mask_np[i][j][k])
                tmp_brain = int(brain_np[i][j][k])
                tmp_contour = int(contour_np[i][j][k])

                if tmp_brain == 0:

                    if tmp_mask == 0:
                        tn_brain += 1
                    if tmp_mask == 1:
                        fp_brain += 1

                elif tmp_brain == 1:
                    sum_in_brain_35 += tmp_seuil_35

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
                    sum_in_contour_35 += tmp_seuil_35

                    if tmp_mask == 1:
                        tp_contour += 1
                    elif tmp_mask == 0:
                        fn_contour += 1

    tfp_brain = tp_brain / (fp_brain + tp_brain)
    mttp_contour = 1 - (fp_contour / (fp_contour + tp_contour))
    return tfp_brain, mttp_contour, sum_in_contour_35, sum_in_brain_35
