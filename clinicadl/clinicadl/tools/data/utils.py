# coding: utf8

import numpy as np
from clinicadl.tools.inputs.filename_types import FILENAME_TYPE
import random
from skimage.draw import ellipse
from scipy.ndimage import gaussian_filter


def load_and_check_tsv(tsv_path, caps_dict, output_path):
    import pandas as pd
    from os.path import join
    from clinica.iotools.utils.data_handling import create_subs_sess_list
    from ..deep_learning.data import check_multi_cohort_tsv

    if tsv_path is not None:
        if len(caps_dict) == 1:
            df = pd.read_csv(tsv_path, sep='\t')
            if ('session_id' not in list(df.columns.values)) or (
                    'participant_id' not in list(df.columns.values)):
                raise Exception("the data file is not in the correct format."
                                "Columns should include ['participant_id', 'session_id']")
        else:
            tsv_df = pd.read_csv(tsv_path, sep='\t')
            check_multi_cohort_tsv(tsv_df, "labels")
            df = pd.DataFrame()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.loc[idx, 'cohort']
                cohort_path = tsv_df.loc[idx, 'path']
                cohort_df = pd.read_csv(cohort_path, sep="\t")
                cohort_df['cohort'] = cohort_name
                df = pd.concat([df, cohort_df])
    else:
        df = pd.DataFrame()
        for cohort, caps_path in caps_dict.items():
            create_subs_sess_list(caps_path, output_path,
                                  is_bids_dir=False, use_session_tsv=False)
            cohort_df = pd.read_csv(
                join(output_path, 'subjects_sessions_list.tsv'), sep="\t")
            cohort_df['cohort'] = cohort
            df = pd.concat([df, cohort_df])

    return df


def find_image_path(caps_dict, participant_id, session_id, cohort, preprocessing):
    from os import path

    if cohort not in caps_dict.keys():
        raise ValueError('Cohort names in labels and CAPS definitions do not match.')

    if preprocessing == "t1-linear":
        image_path = path.join(caps_dict[cohort], 'subjects', participant_id, session_id,
                               't1_linear',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    elif preprocessing == "t1-extensive":
        image_path = path.join(caps_dict[cohort], 'subjects', participant_id, session_id,
                               't1', 'spm', 'segmentation', 'normalized_space',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['skull_stripped'] + '.nii.gz')
    else:
        raise ValueError(
            "Preprocessing %s must be in ['t1-linear', 't1-extensive']." %
            preprocessing)

    return image_path


def binary_t1_pgm(im_data):
    """
    :param im_data: probability gray maps
    :return: binarized probability gray maps
    """
    m = im_data > 0.0
    m = m.astype('float32')
    return m


def im_loss_roi_gaussian_distribution(im_data, atlas_to_mask, min_value):
    gm_masked = np.array(im_data, copy=True)
    gm_masked[atlas_to_mask == 0] = 0

    gm_loss = np.array(gm_masked, copy=True)
    idx = np.nonzero(gm_masked)
    coordinates = []
    for i in range(len(idx[0])):
        coordinates.append([idx[0][i], idx[1][i], idx[2][i]])
    length_coordinates = len(coordinates)  # all the non zero values

    # gaussian distribution with std = 0.1 and media = 0
    n = np.random.normal(loc=0.0, scale=0.1, size=length_coordinates)
    max_value = np.min(n)
    n_new = n + abs(max_value)

    n_diff = n_new * 10 + min_value
    for i, coord in enumerate(coordinates):
        x, y, z = coord
        gm_loss[x, y, z] = gm_masked[x, y, z] - \
            n_diff[i] * (gm_masked[x, y, z]) / 100

    normal_region = np.array(im_data, copy=True)
    normal_region[atlas_to_mask > 0] = 0

    im_with_loss_gm_roi = normal_region + gm_loss

    return im_with_loss_gm_roi


def find_borders_of_one_roi(output_path, i, tsv_atlas):
    """

    :param output_path: where I have saved my borders
    :param i: index of the ROI
    :param tsv_atlas: tsv atlas
    :return: mask with 1 where there are the borders of roi 'i' with the other regions
    """
    import nibabel as nib
    import os

    total_mask = []

    border_1 = nib.load(
        os.path.join(
            output_path,
            'borders-' +
            str(i) +
            '.nii')).get_data()
    for j in set(tsv_atlas.label):
        if j not in [0, i]:
            border_2 = nib.load(
                os.path.join(
                    output_path,
                    'borders-' +
                    str(j) +
                    '.nii')).get_data()
            image_sum = border_1 + border_2
            mask = image_sum > 1
            mask = mask.astype('float32')
            total_mask.append(mask)

    first_mask = total_mask[0]
    for l in total_mask[1:]:
        first_mask = first_mask + l
    first_mask[first_mask > 0] = 1

    return first_mask


def find_border_of_prob_gm_and_atlas(output_path, i, tsv_atlas, im_data):
    """

    :param output_path:
    :param i:
    :param tsv_atlas:
    :param im_data:
    :return:
    """
    mask_of_ROI_AAL2 = find_borders_of_one_roi(output_path, i, tsv_atlas)
    gm = binary_t1_pgm(im_data)
    final_mask_for_gaussian = mask_of_ROI_AAL2 + \
        gm  # AAL2 borders between the ROIS + T1w GM
    f_g = final_mask_for_gaussian > 1
    f_g = f_g.astype('float32')

    idx = np.nonzero(f_g)
    coordinates = []
    for i in range(len(idx[0])):
        coordinates.append([idx[0][i], idx[1][i], idx[2][i]])
    return coordinates


def generate_scales(size):
    if size == "large":
        return random.uniform(1, 1.2), random.uniform(1, 1.2)
    elif size == "small":
        return random.uniform(0.8, 0.9), random.uniform(0.8, 0.9)
    else:
        raise NotImplementedError(
            "Size %s was not implemented for variable sizes." % size)


def generate_shepplogan_phantom(img_size, label=0, smoothing=True):
    img = np.zeros((img_size, img_size))
    center = (img_size + 1.) / 2.0
    a = center - 2
    b = center * 2 / 3 - 2

    color = random.uniform(0.4, 0.6)

    if label == 0:
        roi1, roi2 = "large", "large"
    elif label == 1:
        roi1, roi2 = "large", "small"
    elif label == 2:
        roi1, roi2 = "small", "large"
    else:
        raise ValueError("Subtype %i was not implemented." % label)

    # Skull
    rr, cc = ellipse(center, center, a, b, (img_size, img_size))
    img[rr, cc] = 1

    # Brain
    offset = random.uniform(1, img_size / 32)
    rr, cc = ellipse(center + offset / 2, center, a - offset,
                     b - offset, (img_size, img_size))
    img[rr, cc] = 0.2

    # Central
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales("large")
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(center + offset1, center + offset2,
                     b / 6 * scale1, b / 6 * scale2,
                     (img_size, img_size), rotation=phi)
    img[rr, cc] = color

    # ROI 1
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi1)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(center * 0.6 + offset1, center + offset2,
                     b / 3 * scale1, b / 4 * scale2,
                     (img_size, img_size), rotation=phi)
    img[rr, cc] = color

    # ROI 2
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(center * 1.5 + offset1, center + offset2,
                     b / 10 * scale1, b / 10 * scale2,
                     (img_size, img_size), rotation=phi)
    img[rr, cc] = color

    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(center * 1.5 + offset1, center * 1.1 + offset2,
                     b / 10 * scale1, b / 10 * scale2,
                     (img_size, img_size), rotation=phi)
    img[rr, cc] = color

    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(center * 1.5 + offset1, center * 0.9 + offset2,
                     b / 10 * scale1, b / 10 * scale2,
                     (img_size, img_size), rotation=phi)
    img[rr, cc] = color

    # Ventricle 1
    a_roi = a * random.uniform(0.8, 1.2)
    phi = np.random.uniform(-np.pi / 16, np.pi / 16)
    rr, cc = ellipse(center, center * 0.75, a_roi / 3, a_roi / 6,
                     (img_size, img_size), rotation=np.pi / 8 + phi)
    img[rr, cc] = 0.0

    # Ventricle 2
    a_roi = a * random.uniform(0.8, 1.2)
    phi = np.random.uniform(-np.pi / 16, np.pi / 16)
    rr, cc = ellipse(center, center * 1.25, a_roi / 3, a_roi / 6,
                     (img_size, img_size), rotation=-np.pi / 8 + phi)
    img[rr, cc] = 0.0

    # Random smoothing
    if smoothing:
        sigma = random.uniform(0, 1)
        img = gaussian_filter(img, sigma * img_size /
                              100.)  # smoothing of data

    img.clip(0, 1)

    return img
