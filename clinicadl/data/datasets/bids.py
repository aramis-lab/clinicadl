from collections.abc import Sequence
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional, TypeAlias

import torchio as tio
from pydantic import Field, field_validator

from clinicadl.io import Bids, BidsFileType
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

MasksType: TypeAlias = dict[str, PathType | BidsFileType | tuple[Bids, BidsFileType]]


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

    @classmethod
    def _get_class(cls):
        return BidsDataset


class BidsDataset(
    BidsNiftiDataset, HasConfig[BidsDatasetConfig], BidsTypeDatasetWithConfig
):
    """
    Careful with n_samples in columns.

    ``CapsDataset`` is a custom :py:class:`PyTorch Dataset <torch.utils.data.Dataset>` class for working with
    neuroimaging data in :term:`CAPS` format.

    The user specifies the type of data to work on via ``preprocessing``, the (participant, session)
    pairs to work on via ``data``, and the labels (scalars or segmentation masks) associated to the images
    via ``label``.

    ``CapsDataset`` loads the image and the potential label, and put them in a :py:class:`~clinicadl.data.structures.DataPoint`.
    The user can add additional data in this ``DataPoint`` via the arguments ``columns``, to add the values
    of columns of the DataFrame ``data``, and ``masks``, to add masks associated to the image.

    TransformsHandler to apply to the images are passed via the argument ``transforms``.

    .. note::
        More precisely, transforms are applied to the ``DataPoint``. If you need any additional data to compute
        a transform (e.g. a mask for normalization), you can add them to the ``DataPoint`` via the arguments
        ``columns`` or ``masks``.

    With ``CapsDataset``, it is possible to work on the whole images, or on patches or slices extracted from the
    images. This is also specified via the ``transforms`` argument (e.g. ``transforms=TransformsHandler(extraction=Slice())``).

    .. note::
        - Depending on the type of data you are working on (images, patches, or slices), you may not find the same information
          in the output ``DataPoint``. See :py:mod:`clinicadl.transforms.extraction` for more details.
        - The size of the ``CapsDataset`` depends on the type of data you are working on. For example, if you have 10 images with
          100 slices each, and you want to work on slices, the length of your dataset will be :math:`10\\times100=1,000`.
        - To avoid confusion, we will use the term "sample" to refer to the actual element of the images we are working on
          (patch, slice or the whole image).

    Finally, a ``CapsDataset`` works with tensors, so, before manipulating data, NIfTI files must be converted to PyTorch
    ``.pt`` format with :py:func:`~CapsDataset.to_tensors`. If conversion was already performed,
    :py:func:`~CapsDataset.read_tensor_conversion` must be called.


    Parameters
    ----------
    caps_directory : PathType
        Path to the :term:`CAPS` directory containing the neuroimaging data. A string or a :pathlib.Path:`pathlib.Path <>` object.
    preprocessing : Preprocessing, default=T1Linear()
        Description of the preprocessing steps applied to the data. See :py:mod:`clinicadl.data.datatypes` to know supported preprocessings.
    data : Optional[DataType], default=None
        A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of (participant, session)
        pairs to consider, as well as any other relevant information (e.g. the labels for classification or
        regression).\n
        Only (participant, session) pairs in this TSV file will be in the ``CapsDataset``.\n
        If ``None``, all (participant, session) pairs in ``caps_directory`` will be used. Besides, a TSV file
        will be created in ``caps_directory``, with the list of all (participant, session)
        pairs in the directory that have the wanted ``preprocessing``. The name of the created TSV depends on the preprocessing,
        but it will always start with "overview" (e.g. ``overview_t1-linear_cropped.tsv``,
        ``overview_pet-linear_18FFDG_pons2.tsv``).

        .. warning::
            Beware that your ``.tsv`` files inside ``caps_directory`` may be overwritten. A good practice is not
            to name your own TSV files with a name starting with "overview".

    label : Optional[Union[str, Sequence[str]]], default=None
        A potential label related to the image. It can be:

        - For **classification**: a numeric column passed in the argument ``columns``. The column must contain **integers**.
          For multi-class classification, do not one-hot encode your labels, but keep them all in the same column, numbered
          from ``0`` to ``num_classes-1``.
        - For **regression**: a numeric column passed in the argument ``columns``.
          You can also pass a set of columns if you want to do multi-output regression. The column(s) must contain **floats**.
        - For ``segmentation``: a segmentation mask passed in the argument ``mask``.
        - For ``reconstruction`` or ``generation``: ``None``.

    transforms : TransformsHandler, default=TransformsHandler()
        Transformation pipeline to apply to the data during loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.TransformsHandler`.
    columns : Optional[Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]], default=None
        Columns to get in the DataFrame ``data``, and to put in the :py:class:`~clinicadl.data.structures.DataPoint` returned
        by the ``CapsDataset``.\n
        It is passed via:

        - a list of strings (e.g. ``["age", "sex"]``), corresponding to the names of the columns;
        - or a dictionary (e.g. ``{"age": function, "sex": None}``), where the keys are the names of the columns, and the values
          are the functions to apply to the columns. If the function is ``None``, no function will be applied to the column.

        .. note::
            The potential functions applied to the columns are applied to the **whole column**. They must take as input
            a :py:class:`pandas.Series`, and return a :py:class:`pandas.Series`. For example, it useful to convert
            string labels to integer labels for classification.

    masks : Optional[Sequence[Union[str, PathType]]], default=None
        Masks to load and to put in the :py:class:`~clinicadl.data.structures.DataPoint` returned by the ``CapsDataset``.\n
        A mask can be either a suffix (image-specific masks), or a file in the "masks" folder of
        ``caps_directory`` (common masks).\n
        For example, if ``masks=["brain", "leftHippocampus.nii.gz"]``:

        * For the mask ``"brain"``, a suffix is passed. Therefore, it is understood as an image-specific mask.
          If the image is in ``sub-001/ses-M000/t1_linear/sub-001_ses-M000_T1w.nii.gz``, it will look for the mask in
          ``sub-001/ses-M000/t1_linear/sub-001_ses-M000_brain.nii.gz``.\n
        * For ``"leftHippocampus.nii.gz"``, a path is passed. Therefore, it is understood as a mask common
          to all images. So, ``CapsDataset`` will simply get the mask in ``{caps_directory}/masks/leftHippocampus.nii.gz``.

        .. note::
            The name of the mask in the ``DataPoint`` is inferred:

            - if the mask is passed as a suffix (e.g. ``"brain"``), this suffix will be used for the name;
            - if the mask is passed as a path (e.g. ``"leftHippocampus.nii.gz"``), the name of the file without the
              extension will be used for the name (``"leftHippocampus"``).

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
        If for some (participant, session) pairs, the image corresponding to ``preprocessing``
        cannot be found.
    ValueError
        If the label passed in ``label`` was not passed in ``columns`` or ``masks``.
    ValueError
        If ``label`` is a non-numeric column.
    ValueError
        If ``label`` is a mask, but it is not image-specific.
    FileNotFoundError
        If ``masks`` contain paths that do not match any files.
    ValueError
        If an element in ``columns`` or ``masks`` is in {"image", "label", "affine", "participant", "session"},
        which are protected names.

    Examples
    --------
    .. code-block:: text

        Data look like:

        mycaps
        ├── masks
        │   └── leftHippocampus.nii.gz
        ├── data.tsv
        └── subjects
            ├── sub-001
            │   └── ses-M000
            │       └── pet_linear
            │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
            │           └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
                ...
            ...

        The "data.tsv" file looks like:

        participant_id  session_id   age   sex   diagnosis
        sub-001         ses-M000     55.0  M     CN
        sub-001         ses-M003     55.0  M     AD
        sub-002         ses-M000     62.0  F     MCI
        sub-002         ses-M003     62.0  F     AD
        sub-003         ses-M000     67.0  F     CN
        ...

    .. code-block:: python

        from clinicadl.data import datasets, datatypes
        from clinicadl.transforms import TransformsHandler, extraction
        from clinicadl.transforms.config import (
                ZNormalizationConfig,
                MaskConfig,
                RandomFlipConfig,
            )
        import pandas as pd

        # to convert diagnosis to numeric values
        def diagnosis_to_number(column: pd.Series) -> pd.Series:
            encoding = {"CN": 0, "MCI": 1, "AD": 2}
            return column.apply(lambda x: encoding[x])

    Let's build a dataset for multi-class classification, with normalization, masking, and data augmentation.
    For normalization and masking, we need two masks that we define in ``masks``. We also want the age of the
    participants, and we will ask it in ``columns``.

    .. code-block:: python

        dataset = datasets.CapsDataset(
            caps_directory="mycaps",
            preprocessing=datatypes.PETLinear(
                tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
            ),
            data="mycaps/data.tsv",
            transforms=TransformsHandler(
                image_transforms=[
                    ZNormalizationConfig(masking_method="brain"),
                    MaskConfig(masking_method="leftHippocampus"),
                ],
                sample_transforms=[],
                augmentations=[RandomFlipConfig(flip_probability=0.3)],
            ),
            label="diagnosis",
            columns={"age": None, "diagnosis": diagnosis_to_number},
            masks=["brain", "leftHippocampus.nii.gz"],
        )

        dataset.to_tensors()

    .. code-block:: python

        >>> dataset[0]
        DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'brain', 'leftHippocampus', 'age', 'extraction'); images: 3)
        >>> dataset[0]["age"]
        55.0

    Let's build a dataset for segmentation, working on patches:

    .. code-block:: python

        dataset = datasets.CapsDataset(
            caps_directory="mycaps",
            preprocessing=datatypes.PETLinear(
                tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
            ),
            data="mycaps/data.tsv",
            transforms=TransformsHandler(extraction=extraction.Patch(patch_size=32, stride=32)),
            label="brain",
            masks=["brain"],
        )

        dataset.read_tensor_conversion()

    .. code-block:: python

        >>> dataset[0]
        DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'leftHippocampus', 'extraction', 'patch_index', 'patch_size', 'patch_stride'); images: 3)
        >>> dataset[0]["label"]
        LabelMap(shape: (1, 32, 32, 32); spacing: (0.82, 0.80, 0.80); orientation: RAS+; dtype: torch.IntTensor)    # here the label is a mask

    See Also
    --------
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
        masks: Optional[
            dict[str, PathType | BidsFileType | tuple[PathType | Bids, BidsFileType]]
        ] = None,
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
        Converts raw files to tensors (in PyTorch's ``.pt`` format).

        This is a **mandatory step** before using a ``CapsDataset``, as some checks on data will
        be performed before conversion (shape consistency, voxel spacing consistency, etc.),
        and some important attributes of the dataset will be computed (e.g. its length, which
        depends on the number of samples per image).

        Conversion to tensors also significantly **speeds up data loading** during training or
        inference.

        The user has the possibility to store transformed images, i.e. images on which
        image transforms have already been applied (see ``image_transforms`` in :py:class:`clinicadl.transforms.TransformsHandler`).
        This practice will speed up dataloading during training or inference as the images don't have
        to be transformed each time they are loaded. The drawback is that the saved images can't be
        used by a dataset with other image transforms.

        .. note::
            Images are converted to the same coordinate system (:term:`RAS+`).

        Parameters
        ----------
        n_proc : int, default=1
            Number of cores to use to parallelize the conversion.
        ignore_spacing : bool, default=False
            Whether to ignore the check made on voxel spacings. If ``False``, it will make sure that all
            images have the same voxel spacing before converting them.

            .. warning::
                In most medical image applications, all the images should have the same
                voxel spacing. Be sure that you don't care before disabling this check.

            .. note ::
                To resample your images to a common spacing, have a look at :py:class:`~clinicadl.transforms.config.ResampleConfig`.

        shape_warning : bool, default=True
            Whether to raise a warning if some images in the dataset have different shapes.

        conversion_name : Optional[str], default=None
            The name of the tensor conversion. It determines:

            - the location where tensors will be saved in your directory:
              ``.../sub-*/ses-*/{datatype}/tensors/{conversion_name}``;
            - the name of the ``json`` file that will store information on the conversion:
              ``{directory}/tensor_conversion/{conversion_name}.json``.

            If a conversion with this name already exists:

            - if ``overwrite=True``, the old conversion and the associated
              tensors will be overwritten;
            - else, ``to_tensors`` will try to merge the old tensor conversion with the new one if they
              concern the same type of data (same datatype, same transforms applied, etc.), otherwise an error will be raised.

            If ``None``, the conversion name will be inferred, depending on the datatype, but will always start with
            "default".

            For this reason, if you pass ``conversion_name``, it can't start with "default".

            ``conversion_name`` **cannot** be ``None`` if ``save_transforms=True``.

        overwrite : bool, default=False
            Whether to overwrite an old tensor conversion that as the same ``conversion_name``.

        save_transforms : bool, default=False
            Whether to save raw images as tensors (``False``), or images on which were applied image
            transforms (``True``). Saving transformed images will speed up dataloading. However transformed
            images are specific to a sequence of transforms, so they cannot be used by any future dataset.

        check_transforms : bool, default=True
            If a conversion named ``conversion_name`` already exists and ``overwrite=False``, ``to_tensors`` will try to merge the current
            tensor conversion with the old one. ``check_transforms`` determines whether transforms
            will be checked during the merger. If ``True``, ``to_tensors`` will check that current transforms match
            the transforms applied during the old conversions.\n
            ``check_transforms=False`` is useful when you use custom transforms (i.e. transforms not in ``ClinicaDL``),
            which cannot be checked.

            .. note::
                If ``save_transforms=False``, no such check will be performed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match before setting ``check_transforms=False``.

        Raises
        ------
        ValueError
            If ``conversion_name`` starts with "default".
        ValueError
            If ``conversion_name`` is ``None`` and ``save_transforms=True``.
        TensorConversionError
            If a conversion named ``conversion_name`` already exists and the new conversion cannot
            be merged with the old one.
        TensorConversionError
            If images don't have the same voxel spacing across (participant, session) pairs, and
            ``ignore_spacing=False``.
        TensorConversionError
            If some image-specific masks don't have the same shape and affine matrix as the image.

        Notes
        -----
        If ``shape_warning=True``, raises a warning (only once) if some images have different shapes.

        Examples
        --------
        .. code-block:: text

            Data look like:

            mycaps
            ├── masks
            │   └── leftHippocampus.nii.gz
            ├── data.tsv
            └── subjects
                ├── sub-001
                │   └── ses-M000
                │       └── pet_linear
                │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
                │           └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
                    ...
                ...

        .. code-block:: python

            from clinicadl.data import datasets, datatypes

            dataset = datasets.CapsDataset(
                caps_directory="mycaps",
                datatype=datatypes.PETLinear(
                    tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
                ),
                data="mycaps/data.tsv",
                masks=["brain", "leftHippocampus.nii.gz"],
            )

        .. code-block:: python

            >>> dataset.to_tensors()
            # data are now as follows:
            # mycaps
            # ├── tensor_conversion
            # │   └── default_pet-linear_18FAV45_pons2.json
            # ├── masks
            # │   ├── leftHippocampus.nii.gz
            # │   └── tensors
            # │       └── default
            # │           └── leftHippocampus.pt
            # ├── data.tsv
            # └── subjects
            #     ├── sub-001
            #     │   └── ses-M000
            #     │       └── pet_linear
            #     │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
            #     │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
            #     │           └── tensors
            #     │               └── default
            #     │                   └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
            #         ...
            #     ...

        Here ``sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt`` contains the associated
        image as a tensor, as well as the mask "brain".\n
        Here, we didn't pass a ``conversion_name``, so the name of the ``json`` file and the name of the folder where tensors
        are saved are inferred. If you put a ``conversion_name``:

        .. code-block:: python

            >>> dataset.to_tensors(conversion_name="pet_conversion")
            # data are now as follows:
            # mycaps
            # ├── tensor_conversion
            # │   ├── default_pet-linear_18FAV45_pons2.json
            # │   └── pet_conversion.json
            # ├── masks
            # │   ├── leftHippocampus.nii.gz
            # │   └── tensors
            # │       ├── default
            # │       └── pet_conversion
            # │           └── leftHippocampus.pt
            # ├── data.tsv
            # └── subjects
            #     ├── sub-001
            #     │   └── ses-M000
            #     │       └── pet_linear
            #     │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
            #     │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
            #     │           └── tensors
            #     │               ├── default
            #     │               └── pet_conversion
            #     │                   └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
            #         ...
            #     ...

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
