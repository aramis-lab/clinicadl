from __future__ import annotations

from pathlib import Path
from typing import (
    Callable,
    Optional,
    Sequence,
    Union,
)

import pandas as pd

from clinicadl.transforms.handlers import Transforms
from clinicadl.tsvtools.utils import read_data
from clinicadl.utils.typing import DataFrameType, PathType

from ..datatypes import DataType
from ..readers.caps_reader import CapsReader
from .tensor import TensorDataset, TensorDatasetConfig


class CapsDatasetConfig(TensorDatasetConfig):
    """Config class of ``CapsDataset``."""

    @classmethod
    def _get_class(cls) -> type[CapsDataset]:
        """Returns the class associated to this config class."""
        return CapsDataset


class CapsDataset(TensorDataset):
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

    _config_type = CapsDatasetConfig

    def __init__(
        self,
        directory: PathType,
        datatype: DataType,
        data: Optional[DataFrameType] = None,
        label: Optional[Union[str, Sequence[str]]] = None,
        transforms: Transforms = Transforms(),
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ] = None,
        masks: Optional[Sequence[Union[str, PathType]]] = None,
    ):
        self._caps_reader = CapsReader(directory)
        super().__init__(
            directory=directory,
            datatype=datatype,
            data=data,
            label=label,
            transforms=transforms,
            columns=columns,
            masks=masks,
        )

    def _has_datatype(self, participant: str, session: str, datatype: DataType) -> bool:
        try:
            self._caps_reader.get_image_path(participant, session, datatype)
        except RuntimeError:
            return False

        return True

    def _create_df(self) -> pd.DataFrame:
        df_path = self._caps_reader.create_subjects_sessions_tsv(self.config.datatype)

        return read_data(df_path)

    def _get_image_path(self, participant: str, session: str) -> Path:
        return self._caps_reader.get_image_path(
            participant, session, self.config.datatype
        )

    def _get_common_mask_path(self, mask_name: str) -> Path:
        return self._caps_reader.get_common_mask_path(mask_name)
