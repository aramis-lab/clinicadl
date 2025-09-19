# coding: utf8
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset

from clinicadl.dictionary.words import (
    AFFINE,
    FIRST_INDEX,
    IMAGE,
    LABEL,
    LAST_INDEX,
    MASK,
    N_SAMPLES,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import ExtractionMethod, Sample
from clinicadl.tsvtools.utils import read_data
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)
from clinicadl.utils.json import read_json, update_json, write_json
from clinicadl.utils.typing import DataType, PathType

from ..datatypes.preprocessing import Preprocessing, T1Linear
from ..readers.caps_reader import CapsReader
from ..structures import Column, DataPoint, Mask
from ..tensor_conversion import TensorConversion, TensorConversionInfo

logger = getLogger("clinicadl.caps_dataset")


class CapsDataset(Dataset):
    """
    ``CapsDataset`` is a custom :py:class:`PyTorch Dataset <torch.utils.data.Dataset>` class for working with
    neuroimaging data in :term:`CAPS` format.

    The user specifies the type of data to work on via ``preprocessing``, the (participant, session)
    pairs to work on via ``data``, and the labels (scalars or segmentation masks) associated to the images
    via ``label``.

    ``CapsDataset`` loads the image and the potential label, and put them in a :py:class:`~clinicadl.data.structures.DataPoint`.
    The user can add additional data in this ``DataPoint`` via the arguments ``columns``, to add the values
    of columns of the DataFrame ``data``, and ``masks``, to add masks associated to the image.

    Transforms to apply to the images are passed via the argument ``transforms``.

    .. note::
        More precisely, transforms are applied to the ``DataPoint``. If you need any additional data to compute
        a transform (e.g. a mask for normalization), you can add them to the ``DataPoint`` via the arguments
        ``columns`` or ``masks``.

    With ``CapsDataset``, it is possible to work on the whole images, or on patches or slices extracted from the
    images. This is also specified via the ``transforms`` argument (e.g. ``transforms=Transforms(extraction=Slice())``).

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

    transforms : Transforms, default=Transforms()
        Transformation pipeline to apply to the data during loading. The user also specifies here whether to work on images, patches, or slices.
        See :py:class:`clinicadl.transforms.Transforms`.
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
    ClinicaDLTSVError
        If the DataFrame in ``data`` is empty.
    ClinicaDLTSVError
        If the DataFrame in ``data`` does not contain the columns ``"participant_id"``
        and ``"session_id"``.
    ClinicaDLTSVError
        If the DataFrame in ``data`` contains duplicated (``participant_id``, ``session_id``) pairs.
    ClinicaDLConfigurationError
        If for some (participant, session) pairs, the image corresponding to ``preprocessing``
        cannot be found.
    ClinicaDLArgumentError
        If the label passed in ``label`` was not passed in ``columns`` or ``masks``.
    ClinicaDLArgumentError
        If ``label`` is a non-numeric column.
    ClinicaDLArgumentError
        If ``label`` is a mask, but it is not image-specific.
    FileNotFoundError
        If ``masks`` contain paths that do not match any files.
    ClinicaDLArgumentError
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
        from clinicadl.transforms import Transforms, extraction
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
            transforms=Transforms(
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
            transforms=Transforms(extraction=extraction.Patch(patch_size=32, stride=32)),
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

    def __init__(
        self,
        caps_directory: PathType,
        preprocessing: Preprocessing = T1Linear(),
        data: Optional[DataType] = None,
        label: Optional[Union[str, Sequence[str]]] = None,
        transforms: Transforms = Transforms(),
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ] = None,
        masks: Optional[Sequence[Union[str, PathType]]] = None,
    ):
        self.directory = Path(caps_directory)
        self.preprocessing = preprocessing
        self.transforms = transforms
        self.extraction = transforms.extraction

        self.eval_mode = False
        self.caps_reader = CapsReader(caps_directory)

        _df = self._get_df_from_input(data)
        columns = self._read_columns(columns)
        self._df = self._process_columns(_df, columns)
        self.columns = list(columns.keys())

        self.caps_reader.check_preprocessing(
            self.get_participant_session_couples(), self.preprocessing
        )

        self.individual_masks, self.common_masks = self._read_masks(masks)
        self._individual_mask_names = set(mask.name for mask in self.individual_masks)
        self._common_mask_names = set(mask.name for mask in self.common_masks)
        self.label = self._check_label(label)

        self.tensor_conversion: TensorConversion = TensorConversion(self)
        self.common_masks_tensors: list[Mask] = []

        self._dict = self.model_dump(masks)

    @property
    def df(self) -> pd.DataFrame:
        """The DataFrame passed in ``data``, with additional information."""
        return self._df

    @property
    def converted(self) -> bool:
        """Whether tensor conversion was performed."""
        return self.tensor_conversion.completed

    @property
    def _tensor_conversion_info(self) -> Optional[TensorConversionInfo]:
        """Information on tensor conversion."""
        return self.tensor_conversion.get_info() if self.converted else None

    def to_tensors(
        self,
        n_proc: int = 1,
        ignore_spacing: bool = False,
        shape_warning: bool = True,
        conversion_name: Optional[str] = None,
        overwrite: bool = False,
        save_transforms: bool = False,
        check_transforms: bool = True,
    ) -> None:
        """
        Converts NIfTI files to tensors (in PyTorch's ``.pt`` format), the only format that a
        ``CapsDataset`` can manipulate.

        This is a **mandatory step** before using a ``CapsDataset``, as some checks on data will
        be performed before conversion (shape consistency, voxel spacing consistency, etc.),
        and some important attributes of the ``CapsDataset`` will be computed (e.g. its length, which
        depends on the number of samples per image).

        Conversion to tensors also significantly **speeds up data loading** during training or
        inference.

        The user has the possibility to store transformed images, i.e. images on which
        image transforms have already been applied (see ``image_transforms`` in :py:class:`clinicadl.transforms.Transforms`).
        This practice will speed up dataloading during training or inference as the images don't have
        to be transformed each time they are loaded. The drawback is that the saved images can't be
        used by a ``CapsDataset`` with other image transforms.

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
            Whether to raise a warning if some images in the ``CapsDataset`` have different shapes.

        conversion_name : Optional[str], default=None
            The name of the tensor conversion. It determines:

            - the location where tensors will be saved in your :term:`CAPS`:
              ``{caps_directory}/subjects/sub-*/ses-*/{preprocessing}/tensors/{conversion_name}``;
            - the name of the ``json`` file that will store information on the conversion:
              ``{caps_directory}/tensor_conversion/{conversion_name}.json``.

            If a conversion with this name already exists:

            - if ``overwrite=True``, ``CapsDataset`` will overwrite the old conversion and the associated
              tensors;
            - else, ``CapsDataset`` will try to merge the old tensor conversion with the new one if they
              concern the same type of data (same preprocessing, same transforms applied, etc.), otherwise an error will be raised.

            If ``None``, the tensors will be saved in ``{caps_directory}/subjects/sub-*/ses-*/{preprocessing}/tensors/default``,
            and the name of the ``json`` file will be inferred, depending on the preprocessing, but will always start with
            "default".

            For this reason, if you pass ``conversion_name``, it can't start with "default".

            ``conversion_name`` **cannot** be ``None`` if ``save_transforms=True``.

        overwrite : bool, default=False
            Whether to overwrite an old tensor conversion that as the same ``conversion_name``.

        save_transforms : bool, default=False
            Whether to save raw images as tensors (``False``), or images on which were applied image
            transforms (``True``). Saving transformed images will speed up dataloading. However transformed
            images are specific to a sequence of transforms, so they cannot be used by any future ``CapsDataset``.

        check_transforms : bool, default=True
            If a conversion named ``conversion_name`` already exists and ``overwrite=False``, the ``CapsDataset`` will try to merge the current
            tensor conversion with the old one. ``check_transforms`` determines whether transforms
            will be checked during the merger. If ``True``, ``CapsDataset`` will check that current transforms match
            the transforms applied during the old conversions.\n
            ``check_transforms=False`` is useful when you use custom transforms (i.e. transforms not in ``ClinicaDL``),
            which cannot be checked.

            .. note::
                If ``save_transforms=False``, no such check will be performed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match before setting ``check_transforms=False``.

        Raises
        ------
        ClinicaDLArgumentError
            If ``conversion_name`` starts with "default".
        ClinicaDLArgumentError
            If ``conversion_name`` is ``None`` and ``save_transforms=True``.
        ClinicaDLArgumentError
            If a conversion named ``conversion_name`` already exists and the new conversion cannot
            be merged with the old one.
        ClinicaDLTensorConversionError
            If images don't have the same voxel spacing across (participant, session) pairs, and
            ``ignore_spacing=False``.
        ClinicaDLTensorConversionError
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
                preprocessing=datatypes.PETLinear(
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
        self.tensor_conversion.convert_to_tensors(
            n_proc=n_proc,
            ignore_spacing=ignore_spacing,
            shape_warning=shape_warning,
            conversion_name=conversion_name,
            overwrite=overwrite,
            save_transforms=save_transforms,
            check_transforms=check_transforms,
        )
        self._load_pt_masks()
        self._count_samples()

    def read_tensor_conversion(
        self,
        conversion_name: Optional[str] = None,
        check_transforms: bool = True,
        load_also: Optional[list[str]] = None,
    ) -> None:
        """
        To read an old tensor conversion.

        The function will check that the old conversion works with the current ``CapsDataset``,
        i.e. that the images of the current (participant, session) pairs have been converted
        to tensors, as well as the potential masks.

        If transformed images have been saved, it will also check that the transforms
        applied before conversion match the image transforms of the current ``CapsDataset``,
        unless ``check_transforms=False``.

        See :py:meth:`~CapsDataset.to_tensors` for more information on
        conversion to tensors.

        Parameters
        ----------
        conversion_name : Optional[str], default=None
            The name of the tensor conversion to read. This is what you passed to :py:meth:`~CapsDataset.to_tensors`
            during conversion. If ``None``, it will read the tensors in
            ``{caps_directory}/subjects/sub-*/ses-*/{preprocessing}/tensors/default``.
        check_transforms : bool, default=True
            Whether to check if the image transforms potentially applied before tensor conversion
            match the current ones. ``check_transforms=False`` is useful when you use custom transforms (i.e. transforms
            not in ``ClinicaDL``), which cannot be read by ``ClinicaDL`` and thus cannot be checked.

            .. note::
                If :py:meth:`~CapsDataset.to_tensors` was run with ``save_transforms=False``, no check will
                be performed as the tensors saved have not been transformed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match before setting ``check_transforms=False``.

        load_also : list[str], default=[]
            To load additional information potentially stored in ``.pt`` files. By default, only the image and the masks
            mentioned in the argument ``masks`` of the ``CapsDataset`` will be loaded.

        Raises
        ------
        FileNotFoundError
            If there is no conversion named ``conversion_name``.
        ClinicaDLTensorConversionError
            If the conversion mentioned doesn't work with the
            current ``CapsDataset`` (not the same preprocessing, images not all converted, transforms
            mismatch, etc.).
        ClinicaDLArgumentError
            If an element of ``load_also`` was already passed in the arguments ``columns`` or ``masks``
            of the ``CapsDataset``.

        Examples
        --------
        .. code-block:: text

            Data look like:

            mycaps
            ├── tensor_conversion
            │   ├── default_pet-linear_18FAV45_pons2.json
            │   └── pet_conversion.json
            ├── masks
            │   ├── leftHippocampus.nii.gz
            │   └── tensors
            │       ├── default
            │       └── pet_conversion
            │           └── leftHippocampus.pt
            ├── data.tsv
            └── subjects
                ├── sub-001
                │   └── ses-M000
                │       └── pet_linear
                │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_brain.nii.gz
                │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
                │           └── tensors
                │               ├── default
                │               └── pet_conversion
                │                   └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
                    ...
                ...

        .. code-block:: python

            from clinicadl.data import datasets, datatypes

            dataset = datasets.CapsDataset(
                caps_directory="mycaps",
                preprocessing=datatypes.PETLinear(
                    tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
                ),
                data="mycaps/data.tsv",
                masks=["brain", "leftHippocampus.nii.gz"],
            )

        To read the default conversion:

        .. code-block:: python

            >>> dataset.read_tensor_conversion()

        To read a specific conversion:

        .. code-block:: python

            >>> dataset.read_tensor_conversion(conversion_name="pet_conversion")

        See Also
        --------
        :py:meth:`~CapsDataset.to_tensors`
        """
        if load_also:
            for name in load_also:
                if name in self.columns:
                    raise ClinicaDLArgumentError(
                        f"Cannot load the element '{name}', as you already pass this name in 'columns'."
                    )
                elif name in self._individual_mask_names.union(self._common_mask_names):
                    raise ClinicaDLArgumentError(
                        f"Cannot load the element '{name}', as you already pass this name in 'masks'."
                    )
        self.tensor_conversion.read_conversion(
            conversion_name=conversion_name,
            check_transforms=check_transforms,
            load_also=load_also,
        )
        self._load_pt_masks()
        self._count_samples()

    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        It disables data augmentation in the transformation pipeline.
        """
        self.eval_mode = True

    def train(self) -> None:
        """
        Sets the dataset to training mode.

        It enables data augmentation in the transformation pipeline.
        """
        self.eval_mode = False

    def subset(self, data: DataType) -> CapsDataset:
        """
        To get a subset of the ``CapsDataset`` from a list of (participant, session) pairs.

        Parameters
        ----------
        data : DataType
            A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of (participant, session)
            pairs to extract. This list must be passed via two columns named ``"participant_id"``
            and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        CapsDataset
            A subset of the original ``CapsDataset``, restricted to the (participant, session) pairs mentioned in ``data``.

        Raises
        ------
        ClinicaDLTSVError
            If the DataFrame associated to ``data`` does not contain the columns ``"participant_id"``
            and ``"session_id"``.
        ClinicaDLCAPSError
            If no (participant, session) pairs mentioned in ``data`` are in the current CapsDataset
            (this would lead to an empty dataset).
        """
        new_df = read_data(data, check_protected_names=False).set_index(
            [PARTICIPANT_ID, SESSION_ID]
        )
        df = self._df.set_index([PARTICIPANT_ID, SESSION_ID])
        subset_df = df.loc[new_df.index.intersection(df.index)].reset_index()

        if len(subset_df) == 0:
            raise ClinicaDLCAPSError(
                "No (participant, session) pairs mentioned in 'data' are in the CapsDataset. This would lead to an empty dataset!"
            )

        dataset = deepcopy(self)
        dataset._df = subset_df
        if self.converted:
            self._map_indices_to_images(dataset.df)

        return dataset

    def describe(self) -> Dict[str, Any]:
        """
        Returns a description of the ``CapsDataset``.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
                - ``total_samples``: the size of the dataset, i.e. the total number of samples;
                - ``participant_session_pairs``: the list of (participant, session) pairs in the dataset;
                - ``preprocessing``: the preprocessing parameters;
                - ``extraction``: the extraction parameters.

        Raises
        ------
        ClinicaDLCAPSError
            If slices or patches are extracted from the images, and ``to_tensors`` or
            ``read_tensor_conversion`` has not been run previously (some
            attributes of the ``CapsDataset``, such as its length, depends on the
            number of samples per image).
        """
        return {
            "total_samples": len(self),
            "participant_session_pairs": self.get_participant_session_couples(),
            "preprocessing": self.preprocessing.model_dump(),
            "extraction": self.extraction.model_dump(),
        }

    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample. The information will
        correspond to the information on the image the sample was extracted
        from.

        Parameters
        ----------
        idx : int
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of the DataFrame containing
            the metadata, which is equal to ``data`` if ``data`` was passed when instantiating the
            ``CapsDataset``. If ``data`` was not passed, the only accessible columns are
            ``"participant_id"`` and ``"session_id"``.

        Returns
        -------
        Any
            The information (e.g. the age, the sex, etc.)

        Raises
        ------
        IndexError
            If ``idx`` is not a non-negative integer, greater or equal to
            the length of the dataset.
        KeyError
            If ``column`` is not in the metadata DataFrame.
        ClinicaDLCAPSError
            If slices or patches are extracted from the images and ``to_tensors`` or
            ``read_tensor_conversion`` has not been run previously (some
            attributes of the ``CapsDataset``, such as its length, depends on the
            number of samples per image).
        """
        if not isinstance(idx, int) or idx < 0:
            raise IndexError(f"Index must be a non-negative integer, got {idx}.")
        if idx >= len(self):
            raise IndexError(
                f"Index out of range, there are only {len(self)} samples in the dataset."
            )
        if column not in self._df.columns:
            raise KeyError(
                f"No column named '{column}' in the metadata DataFrame. Present columns are: "
                f"{list(self._df.columns)}"
            )

        row = self._df[(self._df[FIRST_INDEX] <= idx) & (idx <= self._df[LAST_INDEX])]
        return row[column].iloc[0]

    def get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all (participant, session) pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            The list of (participant, session).
        """
        return list(zip(self._df[PARTICIPANT_ID], self._df[SESSION_ID]))

    def __len__(self) -> int:
        """
        Computes the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset, i.e. the number of images
            times the number of samples per image.

        Raises
        ------
        ClinicaDLCAPSError
            If slices or patches are extracted from the images and ``to_tensors`` or
            ``read_tensor_conversion`` has not been run previously (the length depends on the
            number of samples per image).
        """
        if N_SAMPLES not in self._df.columns:
            self._count_samples()
        return int(self._df[N_SAMPLES].sum())

    def __getitem__(self, idx: int) -> Sample:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        Sample
            A structured output containing the processed data and metadata, as a
            :py:class:`~clinicadl.transforms.extraction.Sample`.

        Raises
        ------
        ClinicaDLCAPSError
            If 'to_tensors' or 'read_tensor_conversion' has not been called previously.
        IndexError
            If 'idx' is not an non-negative integer.
        IndexError
            If 'idx' is greater or equal to the length of the dataset.
        FileNotFoundError
            If the '.pt' file cannot be found for the (participant, session) associated
            to 'idx'.
        """
        if not self.converted:
            raise ClinicaDLCAPSError(
                "Cannot find tensor files. Please convert your CapsDataset "
                "to tensors using 'to_tensors', or use 'read_tensor_conversion' if it has "
                "already be done."
            )

        participant, session, sample_index = self._get_sample_meta_data(idx)
        data = self._get_data(participant, session)

        if not self._tensor_conversion_info.transforms:  # image transforms not saved
            data = self.transforms.apply_image_transforms(data)

        data = self.transforms.extract_sample(data, sample_index)

        data = self.transforms.apply_sample_transforms(data)

        if not self.eval_mode:
            data = self.transforms.apply_augmentations(data)

        return data

    ### to read user inputs ###
    def _check_label(
        self, label: Optional[Union[str, Sequence[str]]]
    ) -> Optional[Union[Column, list[Column], Mask]]:
        """
        Checks if 'label' is a valid column name (or column names), a valid mask suffix or None.

        Raises
        ------
        ClinicaDLArgumentError
            If 'label' is not a string or None.
        ClinicaDLArgumentError
            If 'label' is not in the columns or the masks passed by the user.
        """
        if isinstance(label, str):
            if label in self._common_mask_names:
                raise ClinicaDLArgumentError(
                    f"A segmentation mask must be specific to each image, but you passed label={label}, which is "
                    "a non image-specific mask."
                )
            elif label in self._individual_mask_names:
                self.individual_masks = [
                    mask for mask in self.individual_masks if mask.name != label
                ]
                return Mask(label)
            elif label not in self.columns:
                raise ClinicaDLArgumentError(
                    f"Got '{label}' for 'label', but there is no such column or mask."
                )

            label = [label]

        if isinstance(label, list):
            labels_list = []
            for lab in label:
                if lab in self.columns:
                    if not pd.api.types.is_numeric_dtype(self._df[lab]):
                        raise ClinicaDLArgumentError(
                            f"'{lab}' was passed in 'label', but this column is not numeric!"
                        )
                    self.columns.remove(lab)
                    labels_list.append(Column(lab))
                else:
                    raise ClinicaDLArgumentError(
                        f"You passed a list in 'label', and this list can only contain columns passed in 'columns'. But got: '{lab}'"
                    )

            if len(labels_list) == 1:
                return labels_list[0]
            else:
                return labels_list

        elif label is None:
            return None

        else:
            raise ClinicaDLArgumentError(
                f"'label' must be a string, a list, or None. Got: {label}"
            )

    def _read_masks(
        self,
        masks: Optional[Sequence[Union[str, PathType]]],
    ) -> tuple[list[Mask], list[Mask]]:
        """
        Reads the masks passed by the user and splits them between common masks
        and image-specific masks.

        Raises
        ------
        ClinicaDLArgumentError
            If 'masks' is not a list, a tuple, or None.
        FileNotFoundError
            If a path is passed for a mask, and this path does not match any file.
        ClinicaDLArgumentError
            If a suffix is passed for a mask, and this suffix is a name
            among {'image', 'label', 'affine', 'participant', 'session'}.
        """
        if masks is None:
            return [], []
        elif not isinstance(masks, (list, tuple)):
            raise ClinicaDLArgumentError(
                f"'masks' should be a list, a tuple, or None, got: {masks}"
            )

        individual_masks: list[Mask] = []
        common_masks: list[Mask] = []
        for mask_name in masks:
            mask = self._read_mask(mask_name)
            if mask.name in {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION}:
                raise ClinicaDLArgumentError(
                    f"Mask cannot be named '{mask.name}'. {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION} "
                    "are protected names."
                )
            if mask.name in self.columns:
                raise ClinicaDLArgumentError(
                    f"Conflict: '{mask.name}' has been passed in 'columns' AND 'masks'!"
                )
            if mask.is_common_mask:
                common_masks.append(mask)
            else:
                individual_masks.append(mask)

        union = [mask.name for mask in common_masks] + [
            mask.name for mask in individual_masks
        ]
        if len(union) != len(set(union)):
            raise ClinicaDLArgumentError(
                f"Duplicated mask names in 'masks' (got masks={masks}). "
                "Beware that if you pass a path in 'masks' (e.g. 'leftHippocampus.nii.gz'), "
                "CapsDataset will name the mask with its file name, without "
                "the extension (e.g. 'leftHippocampus')."
            )

        return individual_masks, common_masks

    def _read_mask(self, mask: PathType) -> Mask:
        """
        Determines if a mask is a common or an individual mask.
        """
        if Path(mask).suffix:  # it is a file
            return Mask(self.caps_reader.get_common_mask_path(mask))
        else:
            return Mask(mask)

    def _get_df_from_input(self, data: Optional[DataType]) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.

        Raises
        ------
        ClinicaDLArgumentError
            If 'data' is not a DataFrame, a path or None.
        ClinicaDLTSVError
            If 'data' is a TSV file that does not exist.
        ClinicaDLTSVError
            If the DataFrame is empty.
        ClinicaDLTSVError
            If the DataFrame does not contain the columns `"participant_id"`
            and `"session_id"`.
        ClinicaDLTSVError
            If the DataFrame contains duplicated (participant_id, session_id) pairs.
        ClinicaDLConfigurationError
            If the data does not match the preprocessing configuration.
        """
        if data is None:
            data = self.caps_reader.create_subjects_sessions_tsv(self.preprocessing)
            logger.info("Creating a TSV file at %s", data)

        if not isinstance(data, (str, Path, pd.DataFrame)):
            raise ClinicaDLArgumentError(
                f"'data' must be a Pandas DataFrame, a path to a TSV file or None. Got {data}"
            )

        df = read_data(data)

        return deepcopy(df)

    @staticmethod
    def _read_columns(
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ],
    ) -> dict[Column, Optional[Callable[[pd.Series], pd.Series]]]:
        """
        Reads 'columns' argument and put it in a uniform format.

        Raises
        ------
        ClinicaDLArgumentError
            If a 'columns' is not a list, a dict, or None.
        """
        if columns is None:
            return dict()
        elif not isinstance(columns, (list, dict)):
            raise ClinicaDLArgumentError(
                f"'columns' must be a list, a dict, or None. Got: {columns}"
            )

        for col in columns:
            if col in {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION}:
                raise ClinicaDLArgumentError(
                    f"A column cannot be named '{col}'. {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION} "
                    "are protected names."
                )

        if isinstance(columns, list):
            return {col: None for col in columns}
        else:
            return columns

    @staticmethod
    def _process_columns(
        df: pd.DataFrame,
        columns: dict[Column, Optional[Callable[[pd.Series], pd.Series]]],
    ) -> pd.DataFrame:
        """
        Processes the DataFrame with encoding functions passed by the user.

        Raises
        ------
        ClinicaDLArgumentError
            If a column passed in 'columns' is note in the DataFrame.
        """
        for column, encoding in columns.items():
            if column not in df.columns:
                raise KeyError(
                    f"'{column}' was passed in 'columns', but there is no such column in the DataFrame "
                    f"you passed in 'data'. Present columns are: {df.columns}"
                )
            if encoding is None:
                continue
            try:
                df[column] = encoding(df[column])
            except Exception as e:
                raise ClinicaDLArgumentError(
                    f"Unable to process the column '{column}' with the function you passed. "
                    "Make sure that this function takes as an input a Pandas Series, and returns a Pandas Series."
                ) from e

        return df

    ### for __getitem__ ###
    def _get_sample_meta_data(self, idx: int) -> Tuple[str, str, int]:
        """
        Retrieves metadata for a given index.
        'idx' is the index of the sample in the dataset.

        Returns
        -------
        tuple
            - participant (str): ID of the participant.
            - session (str): ID of the session.
            - sample_index (int): index of the extracted sample
            in the original image.

        Raises
        ------
        IndexError
            If 'idx' is out of range.
        """
        participant = self.get_sample_info(idx, PARTICIPANT_ID)

        session = self.get_sample_info(idx, SESSION_ID)
        row = self._df.set_index([PARTICIPANT_ID, SESSION_ID]).loc[
            (participant, session)
        ]
        sample_idx = int(idx - row.at[FIRST_INDEX])

        return participant, session, sample_idx

    def _get_info(self, participant: str, session: str, column: str) -> Any:
        """
        Returns the value of a column for a (participant, session).
        """
        return self._df.set_index([PARTICIPANT_ID, SESSION_ID]).at[
            (participant, session), column
        ]

    def _get_data(self, participant: str, session: str) -> DataPoint:
        """
        Gets data relevant to the (participant, session)
        i.e. the image, the masks (individual and common), and any other
        info asked by the user.

        Conversion to tensors must have been performed first.
        """
        pt_path = self.caps_reader.get_tensor_path(
            participant,
            session,
            self.preprocessing,
            conversion_name=self.tensor_conversion.tensor_folder_name,
            check=False,
        )
        images_dict = torch.load(pt_path, weights_only=True)

        # label
        if self.label is None:
            label = None
        elif isinstance(self.label, Mask):
            label_mask = images_dict[self.label.name]
            label = tio.LabelMap(tensor=label_mask, affine=images_dict[AFFINE])
        elif isinstance(self.label, list):
            label = OrderedDict(
                [(lab, self._get_info(participant, session, lab)) for lab in self.label]
            )
        else:
            label = self._get_info(participant, session, self.label)

        data = DataPoint(
            image=tio.ScalarImage(
                tensor=images_dict[IMAGE], affine=images_dict[AFFINE]
            ),
            label=label,
            participant=participant,
            session=session,
            image_path=pt_path,
            preprocessing=self.preprocessing,
        )

        # individual masks
        for mask in self.individual_masks:
            data.add_mask(
                tio.LabelMap(tensor=images_dict[mask.name], affine=images_dict[AFFINE]),
                mask.name,
            )

        # common masks (already loaded)
        for mask in self.common_masks_tensors:
            data.add_mask(mask.get_associated_mask(), mask.name)

        # columns
        for col in self.columns:
            data[col] = self._get_info(participant, session, col)

        # load_also
        load_also = self._tensor_conversion_info.also
        for name in load_also:
            if load_also[name] == IMAGE:
                data.add_image(
                    tio.ScalarImage(
                        tensor=images_dict[name], affine=images_dict[AFFINE]
                    ),
                    name,
                )
            elif load_also[name] == MASK:
                data.add_mask(
                    tio.LabelMap(tensor=images_dict[name], affine=images_dict[AFFINE]),
                    name,
                )
            else:
                data[name] = images_dict[name]

        return data

    ### other utils ###
    def _load_pt_masks(self) -> None:
        """
        Converts nifti masks to the associated tensor masks
        when 'to_tensors' or 'read_tensor_conversion' is called.
        """
        for mask in self.common_masks:
            mask_pt_path = self.caps_reader.path_to_tensor(
                mask.path, conversion_name=self.tensor_conversion.tensor_folder_name
            )
            self.common_masks_tensors.append(Mask(mask_pt_path))

    def _count_samples(self) -> None:
        """
        Gets the number of samples for each image and puts
        it in the dataframe.

        Raises
        ------
        ClinicaDLCAPSError
            If tensors were not converted before.
        """
        if self.extraction.extract_method == ExtractionMethod.IMAGE:
            self._df[N_SAMPLES] = 1
        else:
            if not self.converted:
                raise ClinicaDLCAPSError(
                    "Needs tensors to compute the length of the dataset (which depends "
                    "on the number of samples per image). Please convert your CapsDataset "
                    "to tensors using 'to_tensors', or use 'read_tensor_conversion' if it has "
                    "already be done."
                )
            if self._tensor_conversion_info.shape:  # uniform shape across the dataset
                first_row = self._df.iloc[0]
                participant, session = first_row[PARTICIPANT_ID], first_row[SESSION_ID]
                self._df[N_SAMPLES] = self._get_n_samples(participant, session)
            else:
                for idx, row in self._df.iterrows():
                    participant = row[PARTICIPANT_ID]
                    session = row[SESSION_ID]
                    self._df.at[idx, N_SAMPLES] = self._get_n_samples(
                        participant, session
                    )

        self._map_indices_to_images(self._df)

    def _get_n_samples(self, participant: str, session: str) -> int:
        """
        Gets the number of samples in an image.

        Raises
        ------
        IndexError
            If an error occurred while extracting samples.
        """
        data = self._get_data(participant, session)
        if not self._tensor_conversion_info.transforms:  # image transforms not saved
            data = self.transforms.apply_image_transforms(data)
        try:
            return self.extraction.num_samples_per_image(data)
        except IndexError as exc:
            raise IndexError(
                f"An error occurred while counting samples in images of ({participant}, {session})."
            ) from exc

    @staticmethod
    def _map_indices_to_images(df: pd.DataFrame) -> None:
        """
        To have in the dataframe the last and the first sample index
        corresponding to each image.
        """
        df[FIRST_INDEX] = (df[N_SAMPLES].cumsum().shift(1)).fillna(0).astype(int)
        df[LAST_INDEX] = (df[N_SAMPLES].cumsum() - 1).astype(int)

    def model_dump(self, masks: Optional[list[PathType]]) -> Dict:
        _dict = {}
        _dict["caps_directory"] = self.directory
        _dict["preprocessing"] = self.preprocessing.to_dict()
        # _dict["data"] = self._df for now there is a pb with these two lines
        # _dict["label"] = self.label
        _dict["transforms"] = self.transforms.to_dict()
        _dict["masks"] = masks

        return _dict

    def write_json(self, json_path: PathType, name: Optional[str]) -> None:
        json_path = Path(json_path)

        if name is not None:
            if json_path.is_file():
                update_json(json_path=json_path, new_data={name: self._dict})
            else:
                write_json(json_path=json_path, data={name: self._dict})
        else:
            if json_path.is_file():
                raise ClinicaDLArgumentError(
                    f"File {json_path} already exists. Please provide a name to save the dataset."
                )
            else:
                write_json(json_path=json_path, data=self._dict)

    @classmethod
    def from_json(cls, json_path: PathType) -> CapsDataset:
        json_path = Path(json_path)
        _dict = read_json(json_path=json_path)
        return CapsDataset(**_dict)
