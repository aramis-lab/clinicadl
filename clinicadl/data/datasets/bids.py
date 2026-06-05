from collections.abc import Sequence
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional, TypeAlias

import torchio as tio
from pydantic import Field, field_validator

from clinicadl.io.bids import Bids, BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType, PathType

from ..structures import (
    CommonMask,
    Image,
    IndividualMask,
)
from ..tensors import TensorConversion
from ..utils import DEFAULT_SPATIAL_CHECKS, SpatialCheck
from .bids_utils import (
    BidsNiftiDataset,
    BidsTypeDatasetConfig,
    BidsTypeDatasetWithConfig,
    ColumnsType,
)
from .tensor import TensorDataset

MasksType: TypeAlias = dict[
    str, PathType | BidsFileType | tuple[PathType | Bids, BidsFileType]
]


def _deserialize_masks(serialized_masks: Optional[dict]) -> Optional[MasksType]:
    """
    To read serialized masks.
    """
    if serialized_masks is None:
        return None

    masks = dict()
    for name, mask in serialized_masks.items():
        if isinstance(mask, dict):
            masks[name] = BidsFileType.from_dict(mask)
        elif isinstance(mask, Sequence) and not isinstance(mask, str):
            masks[name] = (Bids.from_dict(mask[0]), BidsFileType.from_dict(mask[1]))
        else:
            masks[name] = mask

    return masks


class BidsDatasetConfig(ObjectConfig["BidsDataset"], BidsTypeDatasetConfig):
    """Config class to check ``BidsDataset`` inputs."""

    bids: Bids = Field(reader=Bids.from_dict)
    file_type: BidsFileType = Field(reader=BidsFileType.from_dict)
    masks: Optional[dict[str, Path | BidsFileType | tuple[Bids, BidsFileType]]] = Field(
        reader=_deserialize_masks
    )

    @field_validator("bids", mode="before")
    @classmethod
    def _convert_to_bids(cls, v: Any) -> Any:
        """
        Convert a path to a ``Bids``.
        """
        if isinstance(v, (str, Path)):
            return Bids(v)
        return v

    @field_validator("masks", mode="before")
    @classmethod
    def _convert_to_bids_(cls, v: Any) -> Any:
        """
        Convert a path to a ``Bids``.
        """
        if isinstance(v, dict):
            for name, value in v.items():
                if isinstance(value, tuple):
                    v[name] = (cls._convert_to_bids(value[0]), value[1])
        return v

    @field_validator("masks", mode="after")
    @classmethod
    def _resolve_path(
        cls, v: Optional[dict[str, Path | BidsFileType | tuple[Bids, BidsFileType]]]
    ) -> Optional[dict[str, Path | BidsFileType | tuple[Bids, BidsFileType]]]:
        if v:
            for name, value in v.items():
                if isinstance(value, Path):
                    v[name] = value.resolve()

        return v

    @classmethod
    def _get_class(cls):
        return BidsDataset


class BidsDataset(
    BidsNiftiDataset, HasConfig[BidsDatasetConfig], BidsTypeDatasetWithConfig
):
    """
    A :py:class:`~clinicadl.data.datasets.Dataset` working with neuroimaging data organized in :term:`BIDS` (or derivative) format.

    The user specifies the path to the :term:`BIDS` directory via ``bids``, the type of data to load via ``file_type``,
    and the (participant, session) pairs to work on via ``data``.

    ``BidsDataset`` loads the image and the potential masks (see ``masks`` argument), and puts them in a :py:class:`~clinicadl.data.structures.Sample`.
    The user can add additional data in this ``Sample`` via the arguments ``columns``.

    Transformations (e.g., preprocessing or data augmentation) can be applied to the loaded data (see ``transforms`` argument).

    With ``BidsDataset``, it is possible to work on the whole images, or on patches or slices extracted from the
    images. This is also specified via the ``transforms`` argument (e.g., ``transforms=TransformsHandler(extraction=Slice())``).

    .. note::
        - The size of the ``BidsDataset`` depends on the type of data you are working on. For example, if you have 10 images with
          100 slices each, and you want to work on slices, the length of your dataset will be :math:`10\\times100=1,000`.
        - To avoid confusion, we will use the term "sample" to refer to the actual element of the images we are working on
          (patch, slice or the whole image).

    Finally, you may be interested in :py:meth:`to_tensors`, that will convert your NIfTI images to tensors (saved in ``.pt`` files). Since opening
    a ``.pt`` file is much faster than opening a NIfTI file, this may speed up data loading.

    Parameters
    ----------
    bids : PathType | Bids
        The :term:`BIDS` (or derivative) directory where the data will be loaded from. Can be passed as a path or directly as :py:class:`~clinicadl.io.bids.Bids`.

    file_type : BidsFileType
        Defines the files to load in the BIDS directory. The :py:class:`~clinicadl.io.bids.BidsFileType` must contain the requirements necessary to select
        only the relevant files.

    data : Optional[DataFrameType], default=None
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the DataFrame) with the list of (participant, session)
        pairs to consider, as well as any other relevant information (e.g. the age of the participants). Only (participant, session)
        pairs mentioned in this TSV file will be in the ``BidsDataset``.

        If ``None``, all (participant, session) pairs in ``bids`` that have the right ``file_type`` will be considered.

        .. warning::
            Be careful if you pass a DataFrame with a column named ``"n_samples"``. ``BidsDataset`` will understand it as the
            number of samples for each (participant, session) pair.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data after loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.

    columns : Optional[ColumnsType], default=None
        Columns to get in the DataFrame ``data`` and to put in the output :py:class:`~clinicadl.data.structures.Sample`.

        Can be passed via:

        - a list of strings (e.g. ``["age", "sex"]``), corresponding to the names of the columns;
        - or a dictionary (e.g. ``{"age": <function>, "sex": None}``), where the keys are the names of the columns, and the values
          are functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it is useful to convert
            string labels to integer labels for classification.

    masks : Optional[MasksType], default=None
        Masks to be loaded along with the images.

        The masks are passed via a dictionary, whose names will be the names given to the masks in the output
        :py:class:`~clinicadl.data.structures.Sample`, and whose values can be:

        - a path (``str`` or :pathlib.Path:`pathlib.Path <>`) to a NIfTI image: the same mask is used for all
          the (participant, session) pairs.
        - a :py:class:`~clinicadl.io.bids.BidsFileType`: the mask is subject- and session-specific and the
          pattern to find the mask in the ``bids`` is given via the ``BidsFileType``.
        - a tuple (PathType | :py:class:`~clinicadl.io.bids.Bids`, :py:class:`~clinicadl.io.bids.BidsFileType`):
          the mask is subject- and session-specific but is not in the same BIDS dataset as the image. So, here
          the BIDS where to look for the mask must be passed in the first element of the tuple.

    Raises
    ------
    DataFrameError
        If the DataFrame in ``data`` is empty.
    DataFrameError
        If the DataFrame in ``data`` does not contain the columns ``"participant_id"``
        and ``"session_id"``.
    DataFrameError
        If the DataFrame in ``data`` contains duplicated (``participant_id``, ``session_id``) pairs.
    RuntimeError
        If for some (participant, session) pairs, an image corresponding to ``file_type``
        cannot be found in ``bids``.
    ValueError
        If a key is used in ``columns`` and ``masks``.
    ValueError
        If a key in ``columns`` or ``masks`` is equal to the name of one of the attributes of :py:class:`~clinicadl.data.structures.Sample`.

    Examples
    --------
    .. code-block:: text

        bids
        ├── dataset_description.json
        ├── metadata.tsv
        ├── sub-001
        │   ├── ses-M000
        │   │   └── anat
        │   │       ├── sub-001_ses-M000_T1w.nii.gz
        │   │       └── sub-001_ses-M000_label-head_mask.nii.gz
        │   ...
        ...
        └── derivatives
            ├── registration
            │   ├── space-MNI152NLin2009cSym_mask.nii.gz
            │   ...
            └── masks
                ├── dataset_description.json
                ├── sub-001
                │   ├── ses-M000
                │   │   └── anat
                │   │       └── sub-001_ses-M000_label-brain_mask.nii.gz
                │   ...
                ...

        The "metadata.tsv" file looks like:

        participant_id  session_id   age   sex   diagnosis
        sub-001         ses-M000     55.0  M     control
        sub-001         ses-M024     57.0  M     control
        sub-002         ses-M000     62.0  F     control
        sub-002         ses-M024     64.0  F     patient
        sub-003         ses-M000     67.0  F     patient
        ...

    .. code-block:: python

        from clinicadl.data.datasets import BidsDataset
        from clinicadl.io.bids import BidsFileType
        from clinicadl.transforms import TransformsHandler, extraction
        from clinicadl.transforms.config import (
            ZNormalizationConfig,
            ResampleConfig,
            RandomFlipConfig,
        )
        import pandas as pd

        # to convert diagnosis to numeric values
        def diagnosis_to_number(column: pd.Series) -> pd.Series:
            encoding = {"control": 0, "patient": 1}
            return column.apply(lambda x: encoding[x])

    .. code-block:: python

        >>> dataset = BidsDataset(
                bids="bids",
                file_type=BidsFileType(
                    data_type="anat",
                    suffix="T1w",
                ),
                data="bids/metadata.tsv",
                columns=["age"],
            )
        >>> dataset[0]
        Sample(Keys: ('age', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1)
        >>> dataset[0].spatial_shape
        (169, 208, 179)    # full image
        >>> len(dataset)
        50    # 50 lines in the metadata.tsv
        >>> dataset[0].participant_id, dataset[0].session_id, dataset[0].age
        'sub-001', 'ses-M000', 55.0

    .. code-block:: python

        >>> dataset = BidsDataset(
                bids="bids",
                file_type=BidsFileType(
                    data_type="anat",
                    suffix="T1w",
                ),
                data="bids/metadata.tsv",
                columns={"age": None, "diagnosis": diagnosis_to_number},
                transforms=TransformsHandler(
                    extraction=extraction.Patch(patch_size=64),
                ),
            )
        >>> dataset[0]["diagnosis"]
        0    # diagnosis is now encoded
        >>> dataset[0].spatial_shape
        (64, 64, 64)    # patches
        >>> len(dataset)
        1800    # 36 patches per image

    .. code-block:: python

        >>>  dataset = BidsDataset(
                bids="bids",
                file_type=BidsFileType(
                    data_type="anat",
                    suffix="T1w",
                ),
                transforms=TransformsHandler(
                    image_transforms=[
                        ResampleConfig(target="mni"),    # masks can be used in transforms
                        ZNormalizationConfig(masking_method="head"),
                    ],
                    augmentations=[RandomFlipConfig(flip_probability=0.5)],
                ),
                masks={
                    "head": BidsFileType(
                        data_type="anat", suffix="mask", with_entities={"label": "head"}
                    ),    # subject- and session-specific mask that is in the same BIDS
                    "brain": (
                        "bids/derivatives/masks",
                        BidsFileType(
                            data_type="anat", suffix="mask", with_entities={"label": "brain"}
                        ),    # subject- and session-specific mask that is in another BIDS
                    ),
                    "mni": "bids/derivatives/registration/space-MNI152NLin2009cSym_mask.nii.gz",    # same mask for all (subject, session)
                },
            )
        >>> dataset[0]
        Sample(Keys: ('head', 'brain', 'mni', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 4)
        >>> len(dataset)
        60    # all the (participant, session) that have T1w images. Not only the ones in metadata.tsv

    See Also
    --------
    :py:class:`~clinicadl.data.datasets.TensorDataset`
    :py:class:`~clinicadl.data.datasets.ConcatDataset`
    :py:class:`~clinicadl.data.datasets.PairedDataset`
    :py:class:`~clinicadl.data.datasets.UnpairedDataset`
    """

    _config_type = BidsDatasetConfig

    def __init__(
        self,
        bids: PathType | Bids,
        file_type: BidsFileType,
        data: Optional[DataFrameType] = None,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        masks: Optional[MasksType] = None,
    ):
        self.config = self._config_type(
            bids=bids,
            file_type=file_type,
            data=data,
            transforms=transforms,
            columns=columns,
            masks=masks,
        )
        super().__init__(
            image=Image(self.config.bids, self.config.file_type),
            data=self.config.data,
            transforms=self.config.transforms,
            columns=self.config.columns,
            masks=self._read_masks(copy(self.config.masks)),
        )

    def _read_masks(
        self,
        masks: Optional[dict[str, PathType | BidsFileType | tuple[Bids, BidsFileType]]],
    ) -> Optional[dict[str, IndividualMask | CommonMask]]:
        """
        Converts masks to the right format.
        """
        if not masks:
            return None

        for name, mask in masks.items():
            if isinstance(mask, BidsFileType):
                masks[name] = IndividualMask(self.config.bids, mask)
            elif isinstance(mask, tuple):
                masks[name] = IndividualMask(mask[0], mask[1])
            else:
                masks[name] = CommonMask(mask)

        return masks

    def to_tensors(
        self,
        conversion_name: Optional[str] = None,
        spatial_checks: Optional[Iterable[str | SpatialCheck]] = DEFAULT_SPATIAL_CHECKS,
        save_transforms: bool = False,
        description: Optional[str] = None,
        overwrite: bool = False,
        check_transforms: bool = True,
        n_proc: int = 1,
    ) -> TensorDataset:
        """
        Converts NifTI files in the current ``BidsDataset`` to tensors (in PyTorch's ``.pt`` format).

        Conversion to tensors may significantly **speed up data loading**.

        The tensors are saved in a BIDS derivative named ``tensors``. The location of this
        folder relative to the original BIDS depends on the type of BIDS (see :py:class:`clinicadl.io.bids.Bids`).
        A ``.json`` file describing the conversion is also saved at the root of ``tensors``, as well
        as other metadata files (see examples).

        Masks will be converted and saved in the same file as the image they are associated with.

        The user has the possibility to save transformed images, i.e. images on which
        image transforms have already been applied (see ``image_transforms`` in :py:class:`clinicadl.transforms.TransformsHandler`).
        This practice will speed up dataloading during training or inference as the images will not have
        to be transformed each time they are loaded.

        .. note::
            Images are converted to the same coordinate system (:term:`RAS+`).

        Parameters
        ----------
        conversion_name : Optional[str], default=None
            The name of the tensor conversion. Must be **alphanumerical**.

            The output tensors and the output ``.json`` file describing the conversion will have ``"conv-<conversion_name>"``
            in their filenames.

            If a conversion with this name already exists:

            - if ``overwrite=True``, the old tensors will be overwritten;
            - else, ``to_tensors`` will try to append the new conversion to the pre-existing tensor conversion if they
              concern the same type of data (same modality, same transforms applied, etc.), otherwise an error will be raised.

            If ``None``, the conversion name will be ``"raw"``.

            ``conversion_name`` **cannot** be ``None`` if ``save_transforms=True``.

        spatial_checks : Optional[Iterable[str | SpatialCheck]], default=("affine", "shape", "global_spacing")
            Potential spatial checks to perform on the images while converting them:

            - ``"spacing"``: checks **intra-sample voxel spacing consistency**, i.e. that all the images and masks
              in the :py:class:`~clinicadl.data.structures.Sample` output by the current dataset have the same voxel spacing.
            - ``"affine"``: checks **intra-sample affine matrix consistency** (so it includes ``"spacing"``).
            - ``"shape"``: checks **intra-sample spatial shape consistency**.
            - ``"global_spacing"``: checks **inter-sample voxel spacing consistency**, i.e. that all the ``Samples``
              in the dataset have the same voxel spacing (so it includes ``"spacing"``).
            - "``global_shape"``: checks **inter-sample spatial shape consistency** (so it includes ``"shape"``).

            If ``None``, no spatial check performed.

        save_transforms : bool, default=False
            Whether to save raw images without transforms as tensors (``False``), or images with the applied
            transforms (``True``).

        description : Optional[str], default=None
            A potential description of the tensor conversion that will be saved in the description ``.json`` file.

        overwrite : bool, default=False
            Whether to overwrite a pre-existing tensor conversion that has the same ``conversion_name``.
            If a conversion named ``conversion_name`` already exists and ``overwrite=False``, ``to_tensors`` will try to append the current
            tensor conversion to the pre-existing one.

        check_transforms : bool, default=True
            ``check_transforms`` determines whether transforms will be checked when appending to a pre-existing conversion. If ``True``,
            ``to_tensors`` will check that the current transforms match the transforms applied during the pre-existing conversion.\n
            ``check_transforms=False`` is useful when you use custom transforms (i.e. transforms not in ``ClinicaDL``),
            which cannot be checked.

            .. note::
                If ``save_transforms=False``, no such check will be performed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match before setting ``check_transforms=False``.

        n_proc : int, default=1
            Number of cores to use to parallelize the conversion.

        Returns
        -------
        TensorDataset
            A ``TensorDataset`` containing the converted tensors.

        Raises
        ------
        ValueError
            If the user passed ``"raw"`` as a ``conversion_name``.
        ValueError
            If ``conversion_name`` is ``None`` and ``save_transforms=True``.
        TensorConversionError
            If a conversion named ``conversion_name`` already exists and the new conversion cannot
            be appended to the pre-existing one.
        RuntimeError
            If some checks in ``spatial_checks`` fail.

        Examples
        --------
        .. code-block:: text

            bids
            ├── dataset_description.json
            ├── metadata.tsv
            ├── sub-001
            │   ├── ses-M000
            │   │   └── anat
            │   │       ├── sub-001_ses-M000_T1w.nii.gz
            │   │       └── sub-001_ses-M000_label-head_mask.nii.gz
            │   ...
            ...
            └── derivatives
                └── registration
                    ├── space-MNI152NLin2009cSym_mask.nii.gz
                    ...

        .. code-block:: python

            from clinicadl.data.datasets import BidsDataset
            from clinicadl.io.bids import BidsFileType
            from clinicadl.transforms import TransformsHandler, extraction
            from clinicadl.transforms.config import ResampleConfig

            dataset = BidsDataset(
                bids="bids",
                file_type=BidsFileType(
                    data_type="anat",
                    suffix="T1w",
                ),
                transforms=TransformsHandler(
                    image_transforms=[
                        ResampleConfig(target="mni"),
                    ],
                ),
                masks={
                    "head": BidsFileType(
                        data_type="anat", suffix="mask", with_entities={"label": "head"}
                    ),
                    "mni": "bids/derivatives/registration/space-MNI152NLin2009cSym_mask.nii.gz",
                },
            )

            tensor_dataset = dataset.to_tensors(
                conversion_name="T1WithMasks",
                save_transforms=True,
            )

        Data are now as follows:

        .. code-block:: text

            bids
            ├── dataset_description.json
            ├── metadata.tsv
            ├── sub-001
            │   ├── ses-M000
            │   │   └── anat
            │   │       ├── sub-001_ses-M000_T1w.nii.gz
            │   │       └── sub-001_ses-M000_label-head_mask.nii.gz
            │   ...
            ...
            └── derivatives
                ├── registration
                │   ├── space-MNI152NLin2009cSym_mask.nii.gz
                │   ...
                └── tensors
                    ├── dataset_description.json
                    ├── conversions.tsv                                                        <- contains the list of all the conversions
                    ├── src-T1w_conv-T1WithMasks_description.json                              <- contains a description of the conversion
                    ├── src-T1w_conv-T1WithMasks_participantsXsessions.tsv                     <- contains the list of the (participant, session) pairs converted
                    ├── sub-001
                    │   ├── ses-M000
                    │   │   └── anat
                    │   │       ├── sub-001_ses-M000_src-T1w_conv-T1WithMasks_tensors.json     <- contains path to the source files
                    │   │       └── sub-001_ses-M000_src-T1w_conv-T1WithMasks_tensors.pt       <- contains the tensors (the transformed image and masks)
                    │   ...
                    ...

        .. code-block:: python

            >>> tensor_dataset[0]
            Sample(Keys: ('head', 'mni', 'file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 3)


        """
        converter = TensorConversion(self)
        conversion = converter.to_tensors(
            conversion_name=conversion_name,
            spatial_checks=spatial_checks,
            save_transforms=save_transforms,
            description=description,
            overwrite=overwrite,
            check_transforms=check_transforms,
            n_proc=n_proc,
        )
        transforms = deepcopy(self.transforms)
        if conversion.transforms:
            transforms.image_transforms = tio.Compose([])

        return TensorDataset(
            conversion.get_json_path(converter.tensors_dir.path),
            data=copy(self.df),
            transforms=transforms,
            columns=copy(self.columns),
            to_load=list(conversion.masks.keys()) + conversion.additional_data,
        )
