# coding: utf8

import random
from copy import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse

from clinicadl.utils.caps_dataset.data import check_multi_cohort_tsv
from clinicadl.utils.clinica_utils import (
    create_subs_sess_list,
    linear_nii,
    pet_linear_nii,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError


def find_file_type(
    preprocessing: str,
    uncropped_image: bool,
    tracer: str,
    suvr_reference_region: str,
) -> Dict[str, str]:
    if preprocessing == "t1-linear":
        file_type = linear_nii("T1w", uncropped_image)
    elif preprocessing == "pet-linear":
        if tracer is None or suvr_reference_region is None:
            raise ClinicaDLArgumentError(
                "`tracer` and `suvr_reference_region` must be defined "
                "when using `pet-linear` preprocessing."
            )
        file_type = pet_linear_nii(tracer, suvr_reference_region, uncropped_image)
    else:
        raise NotImplementedError(
            f"Generation of synthetic data is not implemented for preprocessing {preprocessing}"
        )

    return file_type


def write_missing_mods(output_dir: Path, output_df: pd.DataFrame):
    missing_path = output_dir / "missing_mods"
    missing_path.mkdir(parents=True, exist_ok=True)

    sessions = output_df.session_id.unique()
    for session in sessions:
        session_df = output_df[output_df.session_id == session]
        out_df = copy(session_df[["participant_id"]])
        out_df["synthetic"] = [1] * len(out_df)
        out_df.to_csv(
            missing_path / f"missing_mods_{session}.tsv", sep="\t", index=False
        )


def load_and_check_tsv(
    tsv_path: Path, caps_dict: Dict[str, Path], output_path: Path
) -> pd.DataFrame:

    if tsv_path is not None:
        if len(caps_dict) == 1:
            df = pd.read_csv(tsv_path, sep="\t")
            if ("session_id" not in list(df.columns.values)) or (
                "participant_id" not in list(df.columns.values)
            ):
                raise Exception(
                    "the data file is not in the correct format."
                    "Columns should include ['participant_id', 'session_id']"
                )
            if "cohort" not in list(df.columns.values):
                df["cohort"] = "single"

        else:
            tsv_df = pd.read_csv(tsv_path, sep="\t")
            check_multi_cohort_tsv(tsv_df, "labels")
            df = pd.DataFrame()
            for idx in range(len(tsv_df)):
                cohort_name = tsv_df.loc[idx, "cohort"]
                cohort_path = tsv_df.loc[idx, "path"]
                cohort_df = pd.read_csv(cohort_path, sep="\t")
                cohort_df["cohort"] = cohort_name
                df = pd.concat([df, cohort_df])
    else:
        df = pd.DataFrame()
        for cohort, caps_path in caps_dict.items():
            create_subs_sess_list(
                caps_path, output_path, is_bids_dir=False, use_session_tsv=False
            )
            cohort_df = pd.read_csv(
                output_path / "subjects_sessions_list.tsv", sep="\t"
            )
            cohort_df["cohort"] = cohort
            df = pd.concat([df, cohort_df])

    return df


def binary_t1_pgm(im_data: np.ndarray) -> np.ndarray:
    """
    Args:
        im_data: probability gray maps

    Returns:
        binarized probability gray maps
    """
    m = im_data > 0.0
    m = m.astype("float32")
    return m


def im_loss_roi_gaussian_distribution(
    im_data: np.ndarray, atlas_to_mask: np.ndarray, min_value: float
) -> np.ndarray:
    """
    Create a smooth atrophy in the input image on the region in the mask.
    The value of the atrophy is computed with a Gaussian so it will appear smooth and
    more realistic.

    Args:
        im_data: Input image that will be atrophied (obtained from a nifti file).
        atlas_to_mask: Binary mask of the region to atrophy.
        min_value: Percentage of atrophy between 0 and 100.
    Returns:
        im_with_loss_gm_roi: Image with atrophy in the specified ROI.
    """
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
        gm_loss[x, y, z] = gm_masked[x, y, z] - n_diff[i] * (gm_masked[x, y, z]) / 100

    normal_region = np.array(im_data, copy=True)
    normal_region[atlas_to_mask > 0] = 0

    im_with_loss_gm_roi = normal_region + gm_loss

    return im_with_loss_gm_roi


# Generate SheppLogan
def generate_scales(size):
    if size == "large":
        return random.uniform(1, 1.2), random.uniform(1, 1.2)
    elif size == "small":
        return random.uniform(0.8, 0.9), random.uniform(0.8, 0.9)
    else:
        raise NotImplementedError(
            f"Size {size} was not implemented for variable sizes."
        )


def generate_shepplogan_phantom(
    img_size: int, label: int = 0, smoothing: bool = True
) -> np.ndarray:
    """
    Generate 2D Shepp-Logan phantom with random regions size. Phantoms also
    simulate different kind of AD by generating smaller ROIs.

    Args:
        img_size: Size of the generated image (img_size x img_size).
        label: Take 0 or 1 or 2. Label of the generated image.
            If 0, the ROIs simulate a CN subject.
            If 1, the ROIs simulate type 1 of AD.
            if 2, the ROIs simulate type 2 of AD.
        smoothing: Default True. Apply Gaussian smoothing to the image.
    Returns:
        img: 2D Sheep Logan phantom with specified label.
    """
    img = np.zeros((img_size, img_size))
    center = (img_size + 1.0) / 2.0
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
        raise NotImplementedError(f"Subtype {label} was not implemented.")

    # Skull
    rr, cc = ellipse(center, center, a, b, (img_size, img_size))
    img[rr, cc] = 1

    # Brain
    offset = random.uniform(1, img_size / 32)
    rr, cc = ellipse(
        center + offset / 2, center, a - offset, b - offset, (img_size, img_size)
    )
    img[rr, cc] = 0.2

    # Central
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales("large")
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(
        center + offset1,
        center + offset2,
        b / 6 * scale1,
        b / 6 * scale2,
        (img_size, img_size),
        rotation=phi,
    )
    img[rr, cc] = color

    # ROI 1
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi1)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(
        center * 0.6 + offset1,
        center + offset2,
        b / 3 * scale1,
        b / 4 * scale2,
        (img_size, img_size),
        rotation=phi,
    )
    img[rr, cc] = color

    # ROI 2
    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(
        center * 1.5 + offset1,
        center + offset2,
        b / 10 * scale1,
        b / 10 * scale2,
        (img_size, img_size),
        rotation=phi,
    )
    img[rr, cc] = color

    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(
        center * 1.5 + offset1,
        center * 1.1 + offset2,
        b / 10 * scale1,
        b / 10 * scale2,
        (img_size, img_size),
        rotation=phi,
    )
    img[rr, cc] = color

    offset1 = random.uniform(1, img_size / 32)
    offset2 = random.uniform(1, img_size / 32)
    scale1, scale2 = generate_scales(roi2)
    phi = random.uniform(-np.pi, np.pi)
    rr, cc = ellipse(
        center * 1.5 + offset1,
        center * 0.9 + offset2,
        b / 10 * scale1,
        b / 10 * scale2,
        (img_size, img_size),
        rotation=phi,
    )
    img[rr, cc] = color

    # Ventricle 1
    a_roi = a * random.uniform(0.8, 1.2)
    phi = np.random.uniform(-np.pi / 16, np.pi / 16)
    rr, cc = ellipse(
        center,
        center * 0.75,
        a_roi / 3,
        a_roi / 6,
        (img_size, img_size),
        rotation=np.pi / 8 + phi,
    )
    img[rr, cc] = 0.0

    # Ventricle 2
    a_roi = a * random.uniform(0.8, 1.2)
    phi = np.random.uniform(-np.pi / 16, np.pi / 16)
    rr, cc = ellipse(
        center,
        center * 1.25,
        a_roi / 3,
        a_roi / 6,
        (img_size, img_size),
        rotation=-np.pi / 8 + phi,
    )
    img[rr, cc] = 0.0

    # Random smoothing
    if smoothing:
        sigma = random.uniform(0, 1)
        img = gaussian_filter(img, sigma * img_size / 100.0)  # smoothing of data

    img.clip(0, 1)

    return img


def mask_processing(mask, percentage, sigma):
    inverse_mask = 1 - mask
    inverse_mask[inverse_mask == 0] = 1 - percentage / 100
    gaussian_mask = gaussian_filter(inverse_mask, sigma=sigma)
    return gaussian_mask
