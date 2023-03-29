# coding: utf8

"""
This file generates data for trivial or intractable (random) data for binary classification.
"""
import tarfile
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from clinica.utils.inputs import RemoteFileStructure, clinica_file_reader, fetch_file

from clinicadl.prepare_data.prepare_data_utils import compute_extract_json
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.maps_manager.iotools import check_and_clean, commandline_to_json
from clinicadl.utils.preprocessing import write_preprocessing
from clinicadl.utils.tsvtools_utils import extract_baseline

from .generate_utils import (
    find_file_type,
    generate_shepplogan_phantom,
    im_loss_roi_gaussian_distribution,
    load_and_check_tsv,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate")


def generate_random_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    tsv_path: Optional[Path] = None,
    mean: float = 0,
    sigma: float = 0.5,
    preprocessing: str = "t1-linear",
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    acq_label: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
):
    """
    Generates a random dataset.

    Creates a random dataset for intractable classification task from the first
    subject of the tsv file (other subjects/sessions different from the first
    one are ignored. Degree of noise can be parameterized.

    Parameters
    ----------
    caps_directory: Path
        Path to the (input) CAPS directory.
    output_dir: Path
        Folder containing the synthetic dataset in CAPS format.
    n_subjects: int
        Number of subjects in each class of the synthetic dataset
    tsv_path: Path
        Path to tsv file of list of subjects/sessions.
    mean: float
        Mean of the gaussian noise
    sigma: float
        Standard deviation of the gaussian noise
    preprocessing: str
        Preprocessing performed. Must be in ['t1-linear', 't1-extensive'].
    multi_cohort: bool
        If True caps_directory is the path to a TSV file linking cohort names and paths.
    uncropped_image: bool
        If True the uncropped image of `t1-linear` or `pet-linear` will be used.
    acq_label: str
        name of the tracer when using `pet-linear` preprocessing.
    suvr_reference_region: str
        name of the reference region when using `pet-linear` preprocessing.

    Returns
    -------
    A folder written on the output_dir location (in CAPS format), also a
    tsv file describing this output

    """
    commandline_to_json(
        {
            "output_dir": output_dir,
            "caps_dir": caps_directory,
            "preprocessing": preprocessing,
            "n_subjects": n_subjects,
            "mean": mean,
            "sigma": sigma,
        }
    )
    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)

    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)

    # Create subjects dir
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)

    # Retrieve image of first subject
    participant_id = data_df.loc[0, "participant_id"]
    session_id = data_df.loc[0, "session_id"]
    cohort = data_df.loc[0, "cohort"]

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        preprocessing, uncropped_image, acq_label, suvr_reference_region
    )

    image_path = Path(
        clinica_file_reader(
            [participant_id], [session_id], caps_dict[cohort], file_type
        )[0][0]
    )
    image_nii = nib.load(image_path)
    image = image_nii.get_data()

    # Create output tsv file
    participant_id_list = [f"sub-RAND{i}" for i in range(2 * n_subjects)]
    session_id_list = ["ses-M00"] * 2 * n_subjects
    diagnosis_list = ["AD"] * n_subjects + ["CN"] * n_subjects
    data = np.array([participant_id_list, session_id_list, diagnosis_list])
    data = data.T
    output_df = pd.DataFrame(
        data, columns=["participant_id", "session_id", "diagnosis"]
    )
    output_df["age_bl"] = 60
    output_df["sex"] = "F"
    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    input_filename = image_path.name
    filename_pattern = "_".join(input_filename.split("_")[2::])
    for i in range(2 * n_subjects):
        gauss = np.random.normal(mean, sigma, image.shape)
        participant_id = f"sub-RAND{i}"
        noisy_image = image + gauss
        noisy_image_nii = nib.Nifti1Image(
            noisy_image, header=image_nii.header, affine=image_nii.affine
        )
        noisy_image_nii_path = (
            output_dir / "subjects" / participant_id / "ses-M00" / "t1_linear"
        )

        noisy_image_nii_filename = f"{participant_id}_ses-M00_{filename_pattern}"
        noisy_image_nii_path.mkdir(parents=True, exist_ok=True)
        nib.save(noisy_image_nii, noisy_image_nii_path / noisy_image_nii_filename)

    write_missing_mods(output_dir, output_df)

    logger.info(f"Random dataset was generated at {output_dir}")


def generate_trivial_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    tsv_path: Optional[Path] = None,
    preprocessing: str = "t1-linear",
    mask_path: Optional[Path] = None,
    atrophy_percent: float = 60,
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    acq_label: str = "fdg",
    suvr_reference_region: str = "pons",
):
    """
    Generates a fully separable dataset.

    Generates a dataset, based on the images of the CAPS directory, where a
    half of the image is processed using a mask to occlude a specific region.
    This procedure creates a dataset fully separable (images with half-right
    processed and image with half-left processed)

    Parameters
    ----------
    caps_directory: Path
        Path to the CAPS directory.
    output_dir: Path
        Folder containing the synthetic dataset in CAPS format.
    n_subjects: int
        Number of subjects in each class of the synthetic dataset.
    tsv_path: Path
        Path to tsv file of list of subjects/sessions.
    preprocessing: str
        Preprocessing performed. Must be in ['linear', 'extensive'].
    mask_path: Path
        Path to the extracted masks to generate the two labels.
    atrophy_percent: float
        Percentage of atrophy applied.
    multi_cohort: bool
        If True caps_directory is the path to a TSV file linking cohort names and paths.
    uncropped_image: bool
        If True the uncropped image of `t1-linear` or `pet-linear` will be used.
    acq_label: str
        Name of the tracer when using `pet-linear` preprocessing.
    suvr_reference_region: str
        Name of the reference region when using `pet-linear` preprocessing.

    Returns
    -------
        Folder structure where images are stored in CAPS format.

    Raises
    ------
        IndexError: if `n_subjects` is higher than the length of the TSV file at `tsv_path`.
    """

    from clinicadl.utils.exceptions import DownloadError

    commandline_to_json(
        {
            "output_dir": output_dir,
            "caps_dir": caps_directory,
            "preprocessing": preprocessing,
            "n_subjects": n_subjects,
            "atrophy_percent": atrophy_percent,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)
    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
    data_df = extract_baseline(data_df)

    home = Path.home()
    cache_clinicadl = home / ".cache" / "clinicadl" / "ressources" / "masks"
    url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
    FILE1 = RemoteFileStructure(
        filename="AAL2.tar.gz",
        url=url_aramis,
        checksum="89427970921674792481bffd2de095c8fbf49509d615e7e09e4bc6f0e0564471",
    )
    cache_clinicadl.mkdir(parents=True, exist_ok=True)

    if n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
        )

    if mask_path is None:
        if not (cache_clinicadl / "AAL2").is_dir():
            print("Downloading AAL2 masks...")
            try:
                mask_path_tar = fetch_file(FILE1, cache_clinicadl)
                tar_file = tarfile.open(mask_path_tar)
                print("File: " + mask_path_tar)
                try:
                    tar_file.extractall(cache_clinicadl)
                    tar_file.close()
                    mask_path = cache_clinicadl / "AAL2"
                except RuntimeError:
                    print("Unable to extract downloaded files.")
            except IOError as err:
                print("Unable to download required templates:", err)
                raise DownloadError(
                    """Unable to download masks, please download them
                    manually at https://aramislab.paris.inria.fr/files/data/masks/
                    and provide a valid path."""
                )
        else:
            mask_path = cache_clinicadl / "AAL2"

    # Create subjects dir
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)

    # Output tsv file
    columns = ["participant_id", "session_id", "diagnosis", "age_bl", "sex"]
    output_df = pd.DataFrame(columns=columns)
    diagnosis_list = ["AD", "CN"]

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        preprocessing, uncropped_image, acq_label, suvr_reference_region
    )

    for i in range(2 * n_subjects):
        data_idx = i // 2
        label = i % 2

        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        cohort = data_df.loc[data_idx, "cohort"]
        image_path = Path(
            clinica_file_reader(
                [participant_id], [session_id], caps_dict[cohort], file_type
            )[0][0]
        )
        image_nii = nib.load(image_path)
        image = image_nii.get_data()

        input_filename = image_path.name
        filename_pattern = "_".join(input_filename.split("_")[2::])

        trivial_image_nii_dir = (
            output_dir / "subjects" / f"sub-TRIV{i}" / session_id / preprocessing
        )

        trivial_image_nii_filename = f"sub-TRIV{i}_{session_id}_{filename_pattern}"

        trivial_image_nii_dir.mkdir(parents=True, exist_ok=True)

        atlas_to_mask = nib.load(mask_path / f"mask-{label + 1}.nii").get_data()

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(
            image, atlas_to_mask, atrophy_percent
        )
        trivial_image_nii = nib.Nifti1Image(trivial_image, affine=image_nii.affine)
        trivial_image_nii.to_filename(
            trivial_image_nii_dir / trivial_image_nii_filename
        )

        # Append row to output tsv
        row = [f"sub-TRIV{i}", session_id, diagnosis_list[label], 60, "F"]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])

    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    write_missing_mods(output_dir, output_df)

    logger.info(f"Trivial dataset was generated at {output_dir}")


def generate_shepplogan_dataset(
    output_dir: Path,
    img_size: int,
    labels_distribution: Dict[str, Tuple[float, float, float]],
    extract_json: str = None,
    samples: int = 100,
    smoothing: bool = True,
):
    """
    Creates a CAPS data set of synthetic data based on Shepp-Logan phantom.
    Source NifTi files are not extracted, but directly the slices as tensors.

    Args:
        output_dir: path to the CAPS created.
        img_size: size of the square image.
        labels_distribution: gives the proportions of the three subtypes (ordered in a tuple) for each label.
        extract_json: name of the JSON file in which generation details are stored.
        samples: number of samples generated per class.
        smoothing: if True, an additional random smoothing is performed on top of all operations on each image.
    """

    check_and_clean(output_dir / "subjects")
    commandline_to_json(
        {
            "output_dir": output_dir,
            "img_size": img_size,
            "labels_distribution": labels_distribution,
            "samples": samples,
            "smoothing": smoothing,
        }
    )
    columns = ["participant_id", "session_id", "diagnosis", "subtype"]
    data_df = pd.DataFrame(columns=columns)

    for i, label in enumerate(labels_distribution.keys()):
        for j in range(samples):
            participant_id = f"sub-CLNC{i}{j:04d}"
            session_id = "ses-M00"
            subtype = np.random.choice(
                np.arange(len(labels_distribution[label])), p=labels_distribution[label]
            )
            row_df = pd.DataFrame(
                [[participant_id, session_id, label, subtype]], columns=columns
            )
            data_df = pd.concat([data_df, row_df])

            # Image generation
            slice_path = (
                output_dir
                / "subjects"
                / participant_id
                / session_id
                / "deeplearning_prepare_data"
                / "slice_based"
                / "custom"
                / f"{participant_id}_{session_id}_space-SheppLogan_axis-axi_channel-single_slice-0_phantom.pt"
            )

            slice_dir = slice_path.parent
            slice_dir.mkdir(parents=True, exist_ok=True)

            slice_np = generate_shepplogan_phantom(
                img_size, label=subtype, smoothing=smoothing
            )
            slice_tensor = torch.from_numpy(slice_np).float().unsqueeze(0)
            torch.save(slice_tensor, slice_path)

            image_path = (
                output_dir
                / "subjects"
                / participant_id
                / session_id
                / "shepplogan"
                / f"{participant_id}_{session_id}_space-SheppLogan_phantom.nii.gz"
            )
            image_dir = image_path.parent
            image_dir.mkdir(parents=True, exist_ok=True)
            with image_path.open("w") as f:
                f.write("0")

    # Save data
    data_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    # Save preprocessing JSON file
    preprocessing_dict = {
        "preprocessing": "custom",
        "mode": "slice",
        "use_uncropped_image": False,
        "prepare_dl": True,
        "extract_json": compute_extract_json(extract_json),
        "slice_direction": 2,
        "slice_mode": "single",
        "discarded_slices": 0,
        "num_slices": 1,
        "file_type": {
            "pattern": f"*_space-SheppLogan_phantom.nii.gz",
            "description": "Custom suffix",
            "needed_pipeline": "shepplogan",
        },
    }
    write_preprocessing(preprocessing_dict, output_dir)
    write_missing_mods(output_dir, data_df)

    logger.info(f"Shepplogan dataset was generated at {output_dir}")
