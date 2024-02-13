"""
Produces a tsv file to study all the nii files and perform the quality check.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from clinicadl.utils.clinica_utils import RemoteFileStructure, fetch_file


def extract_metrics(caps_dir: Path, output_dir: Path, group_label):
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # Load eyes segmentation
    home = Path.home()
    cache_clinicadl = home / ".cache" / "clinicadl" / "segmentation"
    url_aramis = "https://aramislab.paris.inria.fr/files/data/template/"
    FILE1 = RemoteFileStructure(
        filename="eyes_segmentation.nii.gz",
        url=url_aramis,
        checksum="56f699c06cafc62ad8bb5b41b188c7c412d684d810a11d6f4cbb441c0ce944ee",
    )

    if not cache_clinicadl.is_dir():
        cache_clinicadl.mkdir(parents=True)

    segmentation_file = cache_clinicadl / FILE1.filename

    if not segmentation_file.is_file():
        try:
            segmentation_file = fetch_file(FILE1, cache_clinicadl)
        except IOError as err:
            raise IOError("Unable to download required eyes segmentation for QC:", err)

    segmentation_nii = nib.load(segmentation_file)
    segmentation_np = segmentation_nii.get_fdata()

    # Get the GM template
    template_path = (
        caps_dir
        / "groups"
        / f"group-{group_label}"
        / "t1"
        / f"group-{group_label}_template.nii.gz"
    )
    template_nii = nib.load(template_path)
    template_np = template_nii.get_fdata()
    template_np = np.sum(template_np, axis=3)
    template_segmentation_np = template_np * segmentation_np

    # Get the data
    filename = output_dir / "QC_metrics.tsv"
    columns = [
        "participant_id",
        "session_id",
        "max_intensity",
        "non_zero_percentage",
        "frontal_similarity",
    ]
    results_df = pd.DataFrame(columns=columns)

    subjects = list((caps_dir / "subjects").iterdir())
    subjects = [
        subject.stem for subject in subjects if str(subject.stem)[:4:] == "sub-"
    ]
    for subject in subjects:
        subject_path = caps_dir / "subjects" / subject
        sessions = list(subject_path.iterdir())
        sessions = [
            session.stem for session in sessions if str(session.stem)[:4:] == "ses-"
        ]
        for session in sessions:
            image_path = (
                subject_path
                / session
                / "t1"
                / "spm"
                / "segmentation"
                / "normalized_space"
                / (
                    subject
                    + "_"
                    + session
                    + "_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
                )
            )
            if image_path.is_file():
                # GM analysis
                image_nii = nib.load(image_path)
                image_np = image_nii.get_fdata()
                image_segmentation_np = image_np * segmentation_np
                eyes_nmi_value = nmi(
                    occlusion1=template_segmentation_np,
                    occlusion2=image_segmentation_np,
                )

                non_zero_percentage = np.count_nonzero(image_np) / image_np.size

                row = [
                    [
                        subject,
                        session,
                        np.max(image_np),
                        non_zero_percentage,
                        eyes_nmi_value,
                    ]
                ]
                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_df.sort_values("max_intensity", inplace=True, ascending=True)
    results_df.to_csv(filename, sep="\t", index=False)


def nmi(occlusion1, occlusion2):
    """Mutual information for joint histogram"""
    # Convert bins counts to probability values
    hist_inter, _, _ = np.histogram2d(occlusion1.ravel(), occlusion2.ravel())
    hist1, _, _ = np.histogram2d(occlusion1.ravel(), occlusion1.ravel())
    hist2, _, _ = np.histogram2d(occlusion2.ravel(), occlusion2.ravel())

    return (
        2
        * _mutual_information(hist_inter)
        / (_mutual_information(hist1) + _mutual_information(hist2))
    )


def _mutual_information(hgram):
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
