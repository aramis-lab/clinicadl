"""
Toy datasets to use in examples and tutorials.
"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from torchio.download import download_and_extract_archive

from clinicadl.data.datasets.bids import MasksType
from clinicadl.io.bids.file_type.base import BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.download import get_clinicadl_cache_dir
from clinicadl.utils.tsvtools import read_df

from .bids import BidsDataset, ColumnsType, DataFrameType, MasksType

__all__ = [
    "BidsStroke",
    "BidsStrokeSmall",
    "BidsDLBS",
    "BidsDLBSSmall",
    "CapsDLBS",
    "BidsNeuroEmo",
]


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
        self._download()
        super().__init__(self.dir, file_type, data, transforms, columns, masks)

    @property
    @abstractmethod
    def download_url(self) -> str:
        """The URL where the dataset can be downloaded."""

    @property
    def dir(self) -> Path:
        """The directory where the dataset is saved."""
        return self._cache_root_dir / type(self).__name__

    @property
    def _cache_root_dir(self) -> Path:
        return get_clinicadl_cache_dir() / "bids"

    def _download(self) -> None:
        """
        Downloads the dataset and puts it in the cache.
        """
        if not self.dir.is_dir():
            download_and_extract_archive(
                self.download_url,
                download_root=self.dir.parent,
                filename=self.dir.name + ".zip",
                remove_finished=True,
            )


class _BidsStroke(BidsExample):
    def __init__(
        self,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        masks: bool = True,
    ):
        self._download()

        super().__init__(
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
            data=self.dir / "participantsXsessions.tsv",
            transforms=transforms,
            columns=columns,
            masks={
                "lesion_mask": (
                    self.dir / "derivatives" / "lesion_masks",
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

        - a list of strings (e.g. ``["col1", "col2"]``), corresponding to the names of the columns (see
          `here <https://openneuro.org/datasets/ds006533/versions/2.0.0/file-display/participants.json>`_ the columns available);
        - or a dictionary (e.g. ``{"col1": <function>, "col2": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    masks : bool, default=True
        Whether to load the lesion masks along with the images. If ``True``, it will be accessible via the key ``"lesion_mask"``.

    Examples
    --------

    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import BidsStroke
        >>> bids = BidsStroke(columns=["age"], masks=True)
        >>> len(bids)
        50
        >>> bids[0]
        Sample(Keys: ('lesion_mask', 'age', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 2)
        >>> bids[0].spacing
        (1.0, 1.0, 1.0)
    """

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsStrokeV1/BidsStroke.zip"


class BidsStrokeSmall(_BidsStroke):
    """
    A small version of :py:class:`~clinicadl.data.datasets.examples.BidsStroke`.

    The dataset is composed of **10 subjects**, with a single session per subject. Image resolution is **2mm isotropic**.

    Parameters
    ----------
    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["col1", "col2"]``), corresponding to the names of the columns (see
          `here <https://openneuro.org/datasets/ds006533/versions/2.0.0/file-display/participants.json>`_ the columns available);
        - or a dictionary (e.g. ``{"col1": <function>, "col2": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    masks : bool, default=True
        Whether to load the lesion masks along with the images. If ``True``, it will be accessible via the key ``"lesion_mask"``.

    Examples
    --------

    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import BidsStrokeSmall
        >>> bids = BidsStrokeSmall(columns=["age"], masks=True)
        >>> len(bids)
        10
        >>> bids[0]
        Sample(Keys: ('lesion_mask', 'age', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 2)
        >>> bids[0].spacing
        (2.0, 2.0, 2.0)
    """

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsStrokeSmallV1/BidsStrokeSmall.zip"


class _DLBS(BidsExample):
    def __init__(
        self,
        pet: bool = False,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
    ):
        self._pet = pet
        self._download()

        super().__init__(
            file_type=BidsFileType(data_type="pet", suffix="pet")
            if self._pet
            else BidsFileType(data_type="anat", suffix="T1w"),
            data=self._get_df(),
            transforms=transforms,
            columns=columns,
        )

    def _get_df(self) -> pd.DataFrame:
        """
        Gets the right DataFrame, depending on the modality.
        """
        df = read_df(self.dir / "metadata.tsv")
        if self._pet:
            return df[~df["AgePETAmy"].isna()].reset_index(drop=True)

        return df[~df["AgeMRI"].isna()].reset_index(drop=True)

    @property
    def _download_pet_version(self) -> bool:
        return not (self._cache_root_dir / type(self).__name__).is_dir() and self._pet


class BidsDLBS(_DLBS):
    """
    A longitudinal :py:class:`~clinicadl.data.datasets.BidsDataset` dataset with T1w or PET images.
    Adapted from `The Dallas Lifespan Brain Study <https://doi.org/10.18112/openneuro.ds004856.v1.3.0>`_.

    The dataset is composed of **30 subjects**, with **up to 3 sessions per subject**. **Image resolution is 1mm isotropic for T1 images**, but
    **not uniform across the dataset for PET images**.

    Parameters
    ----------
    pet : bool
        Whether to load PET data. Otherwise T1w data will be loaded.

        .. note::
            PET data being much lighter than T1 data, loading time is faster.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["col1", "col2"]``), corresponding to the names of the columns (among ``Sex``, ``AgeMRI``, ``AgePETAmy`` and ``HandednessScore``);
        - or a dictionary (e.g. ``{"col1": <function>, "col2": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import BidsDLBS
        >>> bids = BidsDLBS(pet=False, columns=["AgeMRI"])
        >>> len(bids)
        66
        >>> bids[0].image_path
        (PosixPath('..cache/clinicadl/bids/BidsDLBS/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz'),)

    .. code-block:: python

        >>> bids = BidsDLBS(pet=True, columns=["AgePETAmy"])
        >>> len(bids)
        61
        >>> bids[0].image_path
        (PosixPath('..cache/clinicadl/bids/BidsDLBS/sub-1003/ses-wave1/pet/sub-1003_ses-wave1_trc-18FAV45_run-1_pet.nii.gz'),)
    """

    @property
    def download_url(self) -> str:
        if self._pet:
            if self._download_pet_version:
                return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsDLBSPetV1/BidsDLBSPet.zip"

        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsDLBSV1/BidsDLBS.zip"

    @property
    def dir(self) -> Path:
        root_dir = self._cache_root_dir

        if self._download_pet_version:
            return root_dir / (type(self).__name__ + "Pet")

        return self._cache_root_dir / type(self).__name__


class BidsDLBSSmall(BidsDLBS):
    """
    A small version of :py:class:`~clinicadl.data.datasets.examples.BidsDLBS`.

    The dataset is composed of **10 subjects**, with **3 sessions per subject**. **Image resolution is 2mm isotropic for T1 images**, but
    **not uniform across the dataset for PET images**.

    Parameters
    ----------
    pet : bool
        Whether to load PET data. Otherwise T1w data will be loaded.

        .. note::
            PET data being much lighter than T1 data, loading time is faster.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["col1", "col2"]``), corresponding to the names of the columns (among ``Sex``, ``AgeMRI``, ``AgePETAmy`` and ``HandednessScore``);
        - or a dictionary (e.g. ``{"col1": <function>, "col2": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import BidsDLBSSmall
        >>> bids = BidsDLBSSmall(pet=False, columns=["AgeMRI"])
        >>> len(bids)
        30
        >>> bids[0].image_path
        (PosixPath('..cache/clinicadl/bids/BidsDLBSSmall/sub-1003/ses-wave1/anat/sub-1003_ses-wave1_acq-MPRAGE_run-1_T1w.nii.gz'),)

    .. code-block:: python

        >>> bids = BidsDLBSSmall(pet=True, columns=["AgePETAmy"])
        >>> len(bids)
        24
        >>> bids[0].image_path
        (PosixPath('..cache/clinicadl/bids/BidsDLBSSmall/sub-1003/ses-wave1/pet/sub-1003_ses-wave1_trc-18FAV45_run-1_pet.nii.gz'),)
    """

    @property
    def download_url(self) -> str:
        if self._pet:
            if self._download_pet_version:
                return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsDLBSSmallPetV1/BidsDLBSSmallPet.zip"

        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsDLBSSmallV1/BidsDLBSSmall.zip"


class CapsDLBS(_DLBS):
    """
    A :term:`CAPS` version of :py:class:`~clinicadl.data.datasets.examples.BidsDLBS`.

    T1 images have been processed with :clinica:`Clinica's t1-linear pipeline <Pipelines/T1_Linear/>` and
    PET images with the :clinica:`pet-linear pipeline <Pipelines/PET_Linear/>`.

    The dataset is composed of **5 subjects**, with **a single session per subject**. Image resolution is **1mm isotropic** for all images.

    Parameters
    ----------
    pet : bool
        Whether to load PET data. Otherwise T1w data will be loaded.

    cropped : bool, default=True
        Whether to use cropped images returned by Clinica's ``t1-linear`` (169×208×179). Only relevant if ``pet=False``.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the metadata DataFrame and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["col1", "col2"]``), corresponding to the names of the columns (among ``Sex``, ``AgeMRI``, ``AgePETAmy`` and ``HandednessScore``);
        - or a dictionary (e.g. ``{"col1": <function>, "col2": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import CapsDLBS
        >>> caps = CapsDLBS(pet=False, cropped=True, columns=["AgeMRI"])
        >>> len(caps)
        5
        >>> caps[0].image_path
        (PosixPath('..cache/clinicadl/bids/CapsDLBS/subjects/sub-1003/ses-wave1/t1_linear/sub-1003_ses-wave1_acq-MPRAGE_run-1_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'),)
        >>> caps[0].spatial_shape
        (169, 208, 179)

    .. code-block:: python

        >>> caps = CapsDLBS(pet=False, cropped=False, columns=["AgeMRI"])
        >>> caps[0].spatial_shape
        (193, 229, 193)

    .. code-block:: python

        >>> caps = CapsDLBS(pet=True, columns=["AgePETAmy"])
        >>> caps[0].image_path
        (PosixPath('..cache/clinicadl/bids/CapsDLBS/subjects/sub-1003/ses-wave1/pet_linear/sub-1003_ses-wave1_trc-18FAV45_run-1_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz'),)
        >>> caps[0].spatial_shape
        (169, 208, 179)
    """

    def __init__(
        self,
        pet: bool = False,
        cropped: bool = True,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
    ):
        self._pet = pet
        self._download()

        super(_DLBS, self).__init__(
            file_type=BidsFileType(data_type="pet_linear", suffix="pet")
            if self._pet
            else BidsFileType(
                data_type="t1_linear",
                suffix="T1w",
                with_entities={"desc": "Crop"} if cropped else None,
                without_entities={"desc": "Crop"} if not cropped else None,
            ),
            data=self._get_df(),
            transforms=transforms,
            columns=columns,
        )

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/CapsDLBSV1/CapsDLBS.zip"


class BidsNeuroEmo(BidsExample):
    """
    A :py:class:`~clinicadl.data.datasets.BidsDataset` dataset with fMRI images. Adapted from
    `NeuroEmo: An fMRI Dataset for Emotion Recognition <https://doi.org/10.18112/openneuro.ds005700.v1.2.0>`_.

    The dataset is composed of **5 subjects**, with a single session per subject. There are **more than 200 timepoints per image**.

    Parameters
    ----------
    task : str
        The data to load. Among:

        - ``"fe"``: functional data for the emotion task;
        - ``"rest"``: resting-state functional data.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.data.datasets.examples import BidsNeuroEmo
        >>> bids = BidsNeuroEmo(task="rest")
        >>> bids[0].shape
        (250, 96, 96, 38)

    .. code-block:: python

        >>> bids = BidsNeuroEmo(task="fe")
        (200, 128, 128, 36)
    """

    def __init__(
        self,
        task: str,
        transforms: TransformsHandler = TransformsHandler(),
    ):
        super().__init__(
            file_type=BidsFileType(
                data_type="func", with_entities={"task": task}, suffix="bold"
            ),
            transforms=transforms,
        )

    @property
    def download_url(self) -> str:
        return "https://github.com/aramis-lab/clinicadl-data/releases/download/BidsNeuroEmoV1/BidsNeuroEmo.zip"
