# coding: utf8

"""
This file generates data for trivial or intractable (random) data for binary classification.
"""
import tarfile
from collections import namedtuple
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from joblib import Parallel, delayed
from nilearn.image import resample_to_img

from clinicadl.prepare_data.prepare_data_utils import compute_extract_json
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.clinica_utils import (
    RemoteFileStructure,
    clinicadl_file_reader,
    fetch_file,
)
from clinicadl.utils.exceptions import DownloadError
from clinicadl.utils.maps_manager.iotools import check_and_clean, commandline_to_json
from clinicadl.utils.preprocessing import write_preprocessing
from clinicadl.utils.tsvtools_utils import extract_baseline

from .generate_utils import (
    find_file_type,
    generate_shepplogan_phantom,
    im_loss_roi_gaussian_distribution,
    load_and_check_tsv,
    mask_processing,
    write_missing_mods,
)

logger = getLogger("clinicadl.generate")


def generate_random_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    n_proc: int,
    tsv_path: Optional[Path] = None,
    mean: float = 0,
    sigma: float = 0.5,
    preprocessing: str = "t1-linear",
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    tracer: Optional[str] = None,
    suvr_reference_region: Optional[str] = None,
) -> None:
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
    tracer: str
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
            "n_proc": n_proc,
            "mean": mean,
            "sigma": sigma,
        }
    )

    SESSION_ID = "ses-M00"
    AGE_BL_DEFAULT = 60
    SEX_DEFAULT = "F"

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
        preprocessing, uncropped_image, tracer, suvr_reference_region
    )

    image_path = Path(
        clinicadl_file_reader(
            [participant_id], [session_id], caps_dict[cohort], file_type
        )[0][0]
    )
    image_nii = nib.load(image_path)
    image = image_nii.get_fdata()

    # Create output tsv file
    participant_id_list = [f"sub-RAND{i}" for i in range(2 * n_subjects)]
    session_id_list = ["ses-M00"] * 2 * n_subjects
    diagnosis_list = ["AD"] * n_subjects + ["CN"] * n_subjects

    output_df = pd.DataFrame(
        {
            "participant_id": participant_id_list,
            "session_id": session_id_list,
            "diagnosis": diagnosis_list,
        }
    )

    output_df["age_bl"] = AGE_BL_DEFAULT
    output_df["sex"] = SEX_DEFAULT
    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    input_filename = image_path.name
    filename_pattern = "_".join(input_filename.split("_")[2::])

    def create_random_image(subject_id):
        gauss = np.random.normal(mean, sigma, image.shape)
        participant_id = f"sub-RAND{subject_id}"
        noisy_image = image + gauss
        noisy_image_nii = nib.Nifti1Image(
            noisy_image, header=image_nii.header, affine=image_nii.affine
        )
        noisy_image_nii_path = (
            output_dir / "subjects" / participant_id / SESSION_ID / "t1_linear"
        )

        noisy_image_nii_filename = f"{participant_id}_{SESSION_ID}_{filename_pattern}"
        noisy_image_nii_path.mkdir(parents=True, exist_ok=True)
        nib.save(noisy_image_nii, noisy_image_nii_path / noisy_image_nii_filename)

    Parallel(n_jobs=n_proc)(
        delayed(create_random_image)(subject_id) for subject_id in range(2 * n_subjects)
    )

    write_missing_mods(output_dir, output_df)
    logger.info(f"Random dataset was generated at {output_dir}")


def generate_trivial_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    n_proc: int,
    tsv_path: Optional[Path] = None,
    preprocessing: str = "t1-linear",
    mask_path: Optional[Path] = None,
    atrophy_percent: float = 60,
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    tracer: str = "fdg",
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
    tracer: str
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
            "n_proc": n_proc,
            "atrophy_percent": atrophy_percent,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)
    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
    data_df = extract_baseline(data_df)

    if n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
        )

    if mask_path is None:
        home = Path.home()
        cache_clinicadl = home / ".cache" / "clinicadl" / "ressources" / "masks"
        url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/"
        FILE1 = RemoteFileStructure(
            filename="AAL2.tar.gz",
            url=url_aramis,
            checksum="89427970921674792481bffd2de095c8fbf49509d615e7e09e4bc6f0e0564471",
        )
        cache_clinicadl.mkdir(parents=True, exist_ok=True)

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
        preprocessing, uncropped_image, tracer, suvr_reference_region
    )

    def create_trivial_image(subject_id, output_df):
        data_idx = subject_id // 2
        label = subject_id % 2

        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        cohort = data_df.loc[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id], [session_id], caps_dict[cohort], file_type
            )[0][0]
        )
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()

        input_filename = image_path.name
        filename_pattern = "_".join(input_filename.split("_")[2::])

        trivial_image_nii_dir = (
            output_dir
            / "subjects"
            / f"sub-TRIV{subject_id}"
            / session_id
            / preprocessing
        )

        trivial_image_nii_filename = (
            f"sub-TRIV{subject_id}_{session_id}_{filename_pattern}"
        )

        trivial_image_nii_dir.mkdir(parents=True, exist_ok=True)

        path_to_mask = mask_path / f"mask-{label + 1}.nii"
        if path_to_mask.is_file():
            atlas_to_mask = nib.load(path_to_mask).get_fdata()
        else:
            raise ValueError("masks need to be named mask-1.nii and mask-2.nii")

        # Create atrophied image
        trivial_image = im_loss_roi_gaussian_distribution(
            image, atlas_to_mask, atrophy_percent
        )
        trivial_image_nii = nib.Nifti1Image(trivial_image, affine=image_nii.affine)
        trivial_image_nii.to_filename(
            trivial_image_nii_dir / trivial_image_nii_filename
        )

        # Append row to output tsv
        row = [f"sub-TRIV{subject_id}", session_id, diagnosis_list[label], 60, "F"]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])

        return output_df

    results_df = Parallel(n_jobs=n_proc)(
        delayed(create_trivial_image)(subject_id, output_df)
        for subject_id in range(2 * n_subjects)
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)
    write_missing_mods(output_dir, output_df)
    logger.info(f"Trivial dataset was generated at {output_dir}")


def generate_shepplogan_dataset(
    output_dir: Path,
    img_size: int,
    n_proc: int,
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

    for label_id, label in enumerate(labels_distribution.keys()):

        def create_shepplogan_image(subject_id, data_df):
            # for j in range(samples):
            participant_id = f"sub-CLNC{label_id}{subject_id:04d}"
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
            return data_df

        results_df = Parallel(n_jobs=n_proc)(
            delayed(create_shepplogan_image)(subject_id, data_df)
            for subject_id in range(samples)
        )

        data_df = pd.DataFrame()
        for result in results_df:
            data_df = pd.concat([result, data_df])

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


def generate_hypometabolic_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_subjects: int,
    n_proc: int,
    tsv_path: Optional[Path] = None,
    preprocessing: str = "pet-linear",
    pathology: str = "ad",
    anomaly_degree: float = 30,
    sigma: int = 5,
    uncropped_image: bool = False,
):
    """
    Generates a dataset, based on the images of the CAPS directory, where all
    the images are processed using a mask to generate a specific pathology.

    Parameters
    ----------
    caps_directory: Path
        Path to the CAPS directory.
    output_dir: Path
        Folder containing the synthetic dataset in CAPS format.
    n_subjects: int
        Number of subjects in each class of the synthetic dataset.
    n_proc: int
        Number of cores used during the task.
    tsv_path: Path
        Path to tsv file of list of subjects/sessions.
    preprocessing: str
        Preprocessing performed. For now it must be 'pet-linear'.
    pathology: str
        Name of the pathology to generate.
    anomaly_degree: float
        Percentage of pathology applied.
    sigma: int
        It is the parameter of the gaussian filter used for smoothing.
    uncropped_image: bool
        If True the uncropped image of `t1-linear` or `pet-linear` will be used.

    Returns
    -------
    Folder structure where images are stored in CAPS format.


    Raises
    ------
    IndexError: if `n_subjects` is higher than the length of the TSV file at `tsv_path`.
    """

    commandline_to_json(
        {
            "output_dir": output_dir,
            "caps_dir": caps_directory,
            "preprocessing": preprocessing,
            "n_subjects": n_subjects,
            "n_proc": n_proc,
            "pathology": pathology,
            "anomaly_degree": anomaly_degree,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=False)
    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
    data_df = extract_baseline(data_df)

    if n_subjects > len(data_df):
        raise IndexError(
            f"The number of subjects {n_subjects} cannot be higher "
            f"than the number of subjects in the baseline dataset of size {len(data_df)}"
            f"Please add the '--n_subjects' option and re-run the command."
        )
    checksum_dir = {
        "ad": "2100d514a3fabab49fe30702700085a09cdad449bdf1aa04b8f804e238e4dfc2",
        "bvftd": "5a0ad28dff649c84761aa64f6e99da882141a56caa46675b8bf538a09fce4f81",
        "lvppa": "1099f5051c79d5b4fdae25226d97b0e92f958006f6545f498d4b600f3f8a422e",
        "nfvppa": "9512a4d4dc0003003c4c7526bf2d0ddbee65f1c79357f5819898453ef7271033",
        "pca": "ace36356b57f4db73e17c421a7cfd7ae056a1b258b8126534cf65d8d0be9527a",
        "svppa": "44f2e00bf2d2d09b532cb53e3ba61d6087b4114768cc8ae3330ea84c4b7e0e6a",
    }
    home = Path.home()
    cache_clinicadl = home / ".cache" / "clinicadl" / "ressources" / "masks_hypo"
    url_aramis = "https://aramislab.paris.inria.fr/files/data/masks/hypo/"
    FILE1 = RemoteFileStructure(
        filename=f"mask_hypo_{pathology}.nii",
        url=url_aramis,
        checksum=checksum_dir[pathology],
    )
    cache_clinicadl.mkdir(parents=True, exist_ok=True)
    if not (cache_clinicadl / f"mask_hypo_{pathology}.nii").is_file():
        logger.info(f"Downloading {pathology} masks...")
        # mask_path = fetch_file(FILE1, cache_clinicadl)
        try:
            mask_path = fetch_file(FILE1, cache_clinicadl)
        except:
            DownloadError(
                """Unable to download masks, please download them
                manually at https://aramislab.paris.inria.fr/files/data/masks/
                and provide a valid path."""
            )

    else:
        mask_path = cache_clinicadl / f"mask_hypo_{pathology}.nii"

    mask_nii = nib.load(mask_path)

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        preprocessing, uncropped_image, "18FFDG", "cerebellumPons2"
    )

    # Output tsv file
    columns = ["participant_id", "session_id", "pathology", "percentage"]
    output_df = pd.DataFrame(columns=columns)
    participants = [data_df.loc[i, "participant_id"] for i in range(n_subjects)]
    sessions = [data_df.loc[i, "session_id"] for i in range(n_subjects)]
    cohort = caps_directory

    images_paths = clinicadl_file_reader(participants, sessions, cohort, file_type)[0]
    image_nii = nib.load(images_paths[0])

    mask_resample_nii = resample_to_img(mask_nii, image_nii, interpolation="nearest")
    mask = mask_resample_nii.get_fdata()

    mask = mask_processing(mask, anomaly_degree, sigma)

    # Create subjects dir
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)

    def generate_hypometabolic_image(subject_id, output_df):
        image_path = Path(images_paths[subject_id])
        image_nii = nib.load(image_path)
        image = image_nii.get_fdata()
        if image_path.suffix == ".gz":
            input_filename = Path(image_path.stem).stem
        else:
            input_filename = image_path.stem
        input_filename = input_filename.strip("pet")
        hypo_image_nii_dir = (
            output_dir
            / "subjects"
            / participants[subject_id]
            / sessions[subject_id]
            / preprocessing
        )
        hypo_image_nii_filename = (
            f"{input_filename}pat-{pathology}_deg-{int(anomaly_degree)}_pet.nii.gz"
        )
        hypo_image_nii_dir.mkdir(parents=True, exist_ok=True)

        # Create atrophied image
        hypo_image = image * mask
        hypo_image_nii = nib.Nifti1Image(hypo_image, affine=image_nii.affine)
        hypo_image_nii.to_filename(hypo_image_nii_dir / hypo_image_nii_filename)

        # Append row to output tsv
        row = [
            participants[subject_id],
            sessions[subject_id],
            pathology,
            anomaly_degree,
        ]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])
        return output_df

    results_list = Parallel(n_jobs=n_proc)(
        delayed(generate_hypometabolic_image)(subject_id, output_df)
        for subject_id in range(n_subjects)
    )

    output_df = pd.DataFrame()
    for result_df in results_list:
        output_df = pd.concat([result_df, output_df])

    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    write_missing_mods(output_dir, output_df)

    logger.info(
        f"Hypometabolic dataset was generated, with {anomaly_degree} % of dementia {pathology} at {output_dir}."
    )


def generate_artifacts_dataset(
    caps_directory: Path,
    output_dir: Path,
    n_proc: int,
    tsv_path: Optional[str] = None,
    preprocessing: str = "t1-linear",
    multi_cohort: bool = False,
    uncropped_image: bool = False,
    tracer: str = "fdg",
    suvr_reference_region: str = "pons",
    contrast: bool = False,
    gamma: List = [-0.2, -0.05],
    motion: bool = False,
    translation: List = [2, 4],
    rotation: List = [2, 4],
    num_transforms: int = 2,
    noise: bool = False,
    noise_std: List = [5, 15],
):
    """
    Generates a dataset, based on the images of the CAPS directory, where
    all the images are corrupted with a combination of motion, contrast and
    noise artefacts using torchio simulations.
    Args:
        caps_directory:  Path
            Path to the CAPS directory.
        output_dir: Path
            Folder containing the synthetic dataset in CAPS format.
        n_proc: int
            Number of cores used during the task.
        tsv_path: Path
            Path to tsv file of list of subjects/sessions.
        preprocessing: str
            Preprocessing performed. Must be in ['linear', 'extensive'].
        multi_cohort: bool
            If True caps_directory is the path to a TSV file linking cohort names and paths.
        uncropped_image: bool
            If True the uncropped image of `t1-linear` or `pet-linear` will be used.
        tracer: str
            Name of the tracer when using `pet-linear` preprocessing.
        suvr_reference_region: str
            Name of the reference region when using `pet-linear` preprocessing.
        translation: List
            Translation range in mm of simulated movements.
        rotation : List
            Rotation range in degree of simulated movement.
        num_transformes: int
            Number of simulated movements.
        gamma: List
            Gamma range of simulated contrast.
        noise_std: List
            Stadndard deviation of simulated noise.
    Returns:
        Folder structure where images are stored in CAPS format.
    """

    commandline_to_json(
        {
            "output_dir": output_dir,
            "caps_dir": caps_directory,
            "preprocessing": preprocessing,
        }
    )

    # Transform caps_directory in dict
    caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort=multi_cohort)
    # Read DataFrame
    data_df = load_and_check_tsv(tsv_path, caps_dict, output_dir)
    # Create subjects dir
    (output_dir / "subjects").mkdir(parents=True, exist_ok=True)

    # Output tsv file
    columns = ["participant_id", "session_id", "diagnosis"]
    output_df = pd.DataFrame(columns=columns)

    # Find appropriate preprocessing file type
    file_type = find_file_type(
        preprocessing, uncropped_image, tracer, suvr_reference_region
    )
    artifacts_list = []
    if motion:
        artifacts_list.append("motion")
    if contrast:
        artifacts_list.append("contrast")
    if noise:
        artifacts_list.append("noise")

    def create_artifacts_image(data_idx, output_df):
        participant_id = data_df.loc[data_idx, "participant_id"]
        session_id = data_df.loc[data_idx, "session_id"]
        cohort = data_df.loc[data_idx, "cohort"]
        image_path = Path(
            clinicadl_file_reader(
                [participant_id], [session_id], caps_dict[cohort], file_type
            )[0][0]
        )
        input_filename = image_path.name
        filename_pattern = "_".join(input_filename.split("_")[2::])
        subject_name = input_filename.split("_")[:1][0]
        session_name = input_filename.split("_")[1:2][0]

        artif_image_nii_dir = (
            output_dir / "subjects" / subject_name / session_name / preprocessing
        )
        artif_image_nii_dir.mkdir(parents=True, exist_ok=True)

        artifacts_tio = []
        arti_ext = ""
        for artif in artifacts_list:
            if artif == "motion":
                artifacts_tio.append(
                    tio.RandomMotion(
                        degrees=(rotation[0], rotation[1]),
                        translation=(translation[0], translation[1]),
                        num_transforms=num_transforms,
                    )
                )
                arti_ext += "mot-"
            elif artif == "noise":
                artifacts_tio.append(
                    tio.RandomNoise(
                        std=(noise_std[0], noise_std[1]),
                    )
                )
                arti_ext += "noi-"
            elif artif == "contrast":
                artifacts_tio.append(tio.RandomGamma(log_gamma=(gamma[0], gamma[1])))
                arti_ext += "con-"

        if filename_pattern.endswith(".nii.gz"):
            file_suffix = ".nii.gz"
            filename_pattern = Path(Path(filename_pattern).stem).stem
        elif filename_pattern.endswith(".nii"):
            file_suffix = ".nii"
            filename_pattern = Path(filename_pattern).stem

        artif_image_nii_filename = f"{subject_name}_{session_name}_{filename_pattern}_art-{arti_ext[:-1]}{file_suffix}"

        artifacts = tio.transforms.Compose(artifacts_tio)

        artif_image = artifacts(tio.ScalarImage(image_path))
        artif_image.save(artif_image_nii_dir / artif_image_nii_filename)

        # Append row to output tsv
        row = [subject_name, session_name, artifacts_list]
        row_df = pd.DataFrame([row], columns=columns)
        output_df = pd.concat([output_df, row_df])

        return output_df

    results_df = Parallel(n_jobs=n_proc)(
        delayed(create_artifacts_image)(data_idx, output_df)
        for data_idx in range(len(data_df))
    )
    output_df = pd.DataFrame()
    for result in results_df:
        output_df = pd.concat([result, output_df])

    output_df.to_csv(output_dir / "data.tsv", sep="\t", index=False)

    write_missing_mods(output_dir, output_df)

    logger.info(f"Images corrupted with artefacts were generated at {output_dir}")
