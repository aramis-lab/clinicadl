"""
Automatically reject images incorrectly preprocessed by t1-volume (Unified Segmentation) with 3 criterion
1) max < 0.95
2) percentage of non zero values < 15 % or > 50 %
3) frontal similarity of T1 volume with the template < 0.40
"""

from logging import getLogger
from pathlib import Path

import pandas as pd

from clinicadl.quality_check.t1_volume.utils import extract_metrics


def quality_check(caps_dir: Path, output_directory: Path, group_label):
    logger = getLogger("clinicadl.quality_check")
    extract_metrics(
        caps_dir=caps_dir, output_dir=output_directory, group_label=group_label
    )
    logger.info(
        f"Quality check metrics extracted at {output_directory / 'QC_metrics.tsv'}."
    )
    qc_df = pd.read_csv(output_directory / "QC_metrics.tsv", sep="\t")
    rejection1_df = qc_df[qc_df.max_intensity > 0.95]
    rejection1_df.to_csv(output_directory / "pass_step-1.tsv", sep="\t", index=False)
    logger.info(
        f"Number of sessions removed, based on max intensity: {len(qc_df) - len(rejection1_df)}."
    )
    logger.debug(f"{rejection1_df}")

    rejection2_df = rejection1_df[
        (rejection1_df.non_zero_percentage < 0.5)
        & (rejection1_df.non_zero_percentage > 0.15)
    ]
    rejection2_df.to_csv(output_directory / "pass_step-2.tsv", sep="\t", index=False)
    logger.info(
        f"Number of sessions removed based on non-zero voxels: {len(rejection1_df) - len(rejection2_df)}."
    )
    logger.debug(f"{rejection2_df}")

    rejection3_df = rejection2_df[rejection2_df.frontal_similarity > 0.10]
    rejection3_df.to_csv(output_directory / "pass_step-3.tsv", sep="\t", index=False)
    logger.info(
        f"Number of sessions removed based on frontal similarity with DARTEL "
        f"template: {len(rejection2_df) - len(rejection3_df)}."
    )
    logger.debug(f"{rejection3_df}")
