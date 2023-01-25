"""
Automatically reject images incorrectly preprocessed pet-linear (Unified Segmentation) with ? criterion


"""
from logging import getLogger
from os import path

import pandas as pd

from .utils import extract_metrics


def quality_check(caps_dir, output_directory, acq_label, ref_region, threshold, n_proc):

    logger = getLogger("clinicadl.quality_check")

    extract_metrics(
        caps_dir=caps_dir,
        output_dir=output_directory,
        acq_label=acq_label,
        ref_region=ref_region,
        n_proc=n_proc,
    )
    logger.info(
        f"Quality check metrics extracted at {path.join(output_directory, 'QC_metrics.tsv')}."
    )

    qc_df = pd.read_csv(path.join(output_directory, "QC_metrics.tsv"), sep="\t")
    rejection1_df = qc_df[qc_df.ttp_av < threshold]
    rejection1_df.to_csv(
        path.join(output_directory, "data_cleaned_step1.tsv"), sep="\t", index=False
    )

    rejection2_df = rejection1_df[rejection1_df.tfp_ar < threshold]
    rejection2_df.to_csv(
        path.join(output_directory, "data_cleaned_step2.tsv"), sep="\t", index=False
    )

    rejection3_df = rejection2_df[rejection2_df.tfp_up < threshold]
    rejection3_df.to_csv(
        path.join(output_directory, "data_cleaned_step2.tsv"), sep="\t", index=False
    )

    logger.info(
        f"Number of sessions removed based on max intensity: {len(qc_df) - len(rejection1_df)}."
    )
    logger.debug(f"{rejection2_df}")

    # rejection2_df = rejection1_df[
    #     (rejection1_df.non_zero_percentage < 0.5)
    #     & (rejection1_df.non_zero_percentage > 0.15)]

    # rejection2_df.to_csv(
    #     path.join(output_directory, "pass_step-2.tsv"), sep="\t", index=False
    # )
    # logger.info(
    #     f"Number of sessions removed based on non-zero voxels: {len(rejection1_df) - len(rejection2_df)}."
    # )
    # logger.debug(f"{rejection2_df}")

    # rejection3_df = rejection2_df[rejection2_df.frontal_similarity > 0.10]
    # rejection3_df.to_csv(
    #     path.join(output_directory, "pass_step-3.tsv"), sep="\t", index=False
    # )
    # logger.info(
    #     f"Number of sessions removed based on frontal similarity with DARTEL "
    #     f"template: {len(rejection2_df) - len(rejection3_df)}."
    # )
    # logger.debug(f"{rejection3_df}")
