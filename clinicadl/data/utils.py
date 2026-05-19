"""
Other functions to perform various utility task.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import pandas as pd

from clinicadl.io import Bids
from clinicadl.utils.dictionary.utils import SEP
from clinicadl.utils.enum import BaseEnum
from clinicadl.utils.typing import PathType
from clinicadl.utils.variables import SPACING_RTOL

if TYPE_CHECKING:
    from .datasets import Dataset
    from .structures import DataPoint, Sample


def remove_tensors(json_path: PathType) -> None:
    """
    To delete tensors in a dataset.

    Will remove all the tensors saved with :py:meth:`clinicadl.data.datasets.BidsDataset.to_tensors`
    associated with the input ``.json`` file.

    Parameters
    ----------
    json_path : PathType
        Path to the ``.json`` file associated to the tensor conversion you want to delete.

    Examples
    --------

    .. code-block::

        from clinicadl.data.datasets import BidsDataset
        from clinicadl.data.utils import remove_tensors
        from clinicadl.io import BIDSFileType
        from pathlib import Path

        dataset = BidsDataset(
            bids="bids_path", file_type=BIDSFileType(data_type="anat", suffix="T1w")
        )
        dataset.to_tensors()  # the json file is "bids_path/derivatives/tensors/src-T1w_conv-raw_description.json"

    .. code-block::

        >>> remove_tensors("bids_path/derivatives/tensors/src-T1w_conv-raw_description.json")
        >>> Path("bids_path/derivatives/tensors/src-T1w_conv-raw_description.json").is_file()
        False
        >>> Path("bids_path/derivatives/tensors/src-T1w_conv-raw_participantsXsessions.tsv").is_file()
        False
        >>> Path("bids_path/derivatives/tensors/sub-000/ses-M000/sub-000_ses-M000_src-T1w_conv-raw_tensors.pt").is_file()
        False

    See Also
    --------
    :py:meth:`clinicadl.data.datasets.BidsDataset.to_tensors`
    """
    from .tensors.utils import ConversionRow, TensorDescription

    json_path = Path(json_path)

    tensors_dir = Bids(json_path.parent)
    tensor_conversion = TensorDescription.read(json_path)

    pattern = tensors_dir.build_path(tensor_conversion.tensor_type).stem
    for path in tensors_dir.path.rglob("*" + pattern + "*"):
        path.unlink()

    tensor_conversion.get_tsv_path(tensors_dir.path).unlink()
    json_path.unlink()

    conversions_tsv_path = tensor_conversion.get_conversions_tsv_path(tensors_dir.path)
    df = pd.read_csv(conversions_tsv_path, sep=SEP)
    col_name = next(f for f in fields(ConversionRow) if "json" in f.name).name
    print(json_path.name)
    print(df[col_name] != json_path.name)
    df = df[df[col_name] != json_path.name]
    df.to_csv(conversions_tsv_path, sep=SEP, index=False)


class SpatialCheck(str, BaseEnum):
    """
    Possible spatial checks performed to check consistency of images and
    masks in a sample and across a samples.
    """

    SPACING = "spacing"
    AFFINE = "affine"
    SHAPE = "shape"
    GLOBAL_SPACING = "global_spacing"
    GLOBAL_SHAPE = "global_shape"


DEFAULT_SPATIAL_CHECKS = (
    "affine",
    "shape",
    "global_spacing",
)


class DatasetChecker:
    """
    To perform spatial checks on a :py:class:`clinicadl.data.datasets.Dataset`.
    """

    def __init__(
        self,
        spatial_checks: Optional[Iterable[str | SpatialCheck]],
    ):
        self.spatial_checks = (
            [SpatialCheck(check) for check in spatial_checks] if spatial_checks else []
        )
        self.ref_sample: Optional[Sample] = None
        self.enabled = True

    def reset(self):
        """
        Resets the running statistics tracked on the dataset.
        """
        self.ref_sample = None
        self.enabled = True

    def check(self, dataset: Dataset[Sample]) -> None:
        """
        Performs the checks on the input dataset.
        It iterates over all the samples to see if they are loaded correctly,
        and performs spatial checks on the samples, depending on the value of ``self.spatial_checks``.
        """
        self.reset()

        for sample in dataset:
            self.check_data_point(sample)

    def check_data_point(self, data_point: DataPoint) -> None:
        """
        Checks spacing, affine matrix, and/or image shape consistency in a sample.
        Also compares to a reference sample to check consistency across samples.
        """
        if not self.enabled:
            return

        if (
            SpatialCheck.SPACING in self.spatial_checks
            and SpatialCheck.GLOBAL_SPACING not in self.spatial_checks
        ):
            _check_intra_sample_consistency(
                data_point, attr="spacing", desc="voxel spacing"
            )

        if SpatialCheck.AFFINE in self.spatial_checks:
            _check_intra_sample_consistency(
                data_point, attr="affine", desc="affine matrix"
            )

        if (
            SpatialCheck.SHAPE in self.spatial_checks
            and SpatialCheck.GLOBAL_SHAPE not in self.spatial_checks
        ):
            _check_intra_sample_consistency(
                data_point, attr="spatial_shape", desc="spatial shape"
            )

        if self.ref_sample is None:
            self.ref_sample = data_point

        if SpatialCheck.GLOBAL_SPACING in self.spatial_checks:
            _check_dataset_consistency(
                data_point,
                self.ref_sample,
                tolerance=SPACING_RTOL,
                attr="spacing",
                desc="voxel spacing",
            )

        if SpatialCheck.GLOBAL_SHAPE in self.spatial_checks:
            _check_dataset_consistency(
                data_point,
                self.ref_sample,
                tolerance=0,
                attr="spatial_shape",
                desc="spatial shape",
            )


def _check_intra_sample_consistency(data_point: DataPoint, attr: str, desc: str) -> Any:
    """
    Checks that an attribute (e.g., voxel spacing) is consistent
    in a sample.
    """
    try:
        return getattr(data_point, attr)
    except RuntimeError as exc:
        exc.add_note(
            f"\nAn error occurred when checking ({data_point.participant}, {data_point.session}) (see above). "
            f"If you don't care about {desc} consistency and want to ignore this error, please modify 'spatial_checks'."
        )
        raise


def _check_dataset_consistency(
    data_point: DataPoint,
    ref_data_point: Optional[DataPoint],
    tolerance: float,
    attr: str,
    desc: str,
) -> None:
    """
    Checks that a sample attribute (e.g., voxel spacing) is consistent
    with a reference sample.
    """
    attr_value = _check_intra_sample_consistency(data_point, attr, desc)
    ref_attr_value = getattr(ref_data_point, attr)
    if not np.isclose(attr_value, ref_attr_value, rtol=tolerance).all():
        raise RuntimeError(
            f"Different {desc} found in the dataset: "
            f"for example, {desc} is {attr_value} for ({data_point.participant}, {data_point.session}), "
            f"but {ref_attr_value} for ({ref_data_point.participant}, {ref_data_point.session}).\n"
            f"If you don't care about {desc} consistency and want to ignore this error, please modify 'spatial_checks'."
        )
