from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from torchio.download import download_and_extract_archive

from clinicadl.data.datasets.bids import MasksType
from clinicadl.io.bids.file_type.base import BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.download import get_clinicadl_cache_dir

from .bids import BidsDataset, ColumnsType, DataFrameType, MasksType


class BidsExample(BidsDataset, ABC):
    """
    Base class for BIDS examples that download data from an URL.
    """

    def __init__(
        self,
        file_type: BidsFileType,
        data: Optional[DataFrameType] = None,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        masks: Optional[MasksType] = None,
    ):
        super().__init__(self._download(), file_type, data, transforms, columns, masks)

    @property
    @abstractmethod
    def download_url(self) -> str:
        """The URL where the BIDS can be downloaded."""

    def _download(self) -> Path:
        """
        Downloads the dataset and puts it in the cache.
        """
        cache_dir = get_clinicadl_cache_dir() / "bids"
        name = type(self).__name__
        bids_dir = cache_dir / name

        if not bids_dir.is_dir():
            download_and_extract_archive(
                self.download_url,
                download_root=cache_dir,
                filename=name + ".zip",
                remove_finished=True,
            )

        return bids_dir


class _BidsStroke(BidsExample):
    def __init__(
        self,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        masks: bool = True,
    ):
        bids_dir = self._download()
        super().__init__(
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data=bids_dir / "participantsXsessions.tsv",
            transforms=transforms,
            columns=columns,
            masks={
                "lesion_mask": (
                    bids_dir / "derivatives" / "lesion_masks",
                    BidsFileType(data_type="anat", suffix="dseg"),
                )
            }
            if masks
            else None,
        )


class BidsStroke(_BidsStroke):
    """
    A :py:class:`~clinicadl.data.datasets.BidsDataset` dataset with T1w images of left hemisphere chronic stroke survivors
    together with the lesion masks. Adapted from
    `Precise vs. Approximate Numeracy in Stroke Participants <https://doi.org/10.18112/openneuro.ds006533.v2.0.0>`_.

    The dataset is composed of **50 subjects**, with a single session per subject. Image resolution is **1mm isotropic**.

    Parameters
    ----------
    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["age", "sex"]``), corresponding to the names of the columns;
        - or a dictionary (e.g. ``{"age": <function>, "sex": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    masks : bool, default=True
        Whether to load the lesion masks along with the images. If ``True``, it will be accessible via the key ``"lesion_mask"``.
    """

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsStrokev1/BidsStroke.zip"


class BidsStrokeSmall(_BidsStroke):
    """
    A :py:class:`~clinicadl.data.datasets.BidsDataset` dataset with T1w images of left hemisphere chronic stroke survivors
    together with the lesion masks. Adapted from
    `Precise vs. Approximate Numeracy in Stroke Participants <https://doi.org/10.18112/openneuro.ds006533.v2.0.0>`_.

    The dataset is composed of **10 subjects**, with a single session per subject. Image resolution is **2mm isotropic**.

    Parameters
    ----------
    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["age", "sex"]``), corresponding to the names of the columns;
        - or a dictionary (e.g. ``{"age": <function>, "sex": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    masks : bool, default=True
        Whether to load the lesion masks along with the images. If ``True``, it will be accessible via the key ``"lesion_mask"``.
    """

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsStrokeSmallV1/BidsStrokeSmall.zip"
