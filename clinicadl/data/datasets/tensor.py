from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import (
    Any,
    Optional,
)

import torch
import torchio as tio
from pydantic import Field
from typing_extensions import Self

from clinicadl.utils.dictionary.words import (
    AFFINE,
    DF,
    IMAGE,
    MASK,
    TENSOR_CONVERSION,
)

from ..structures.label import Mask
from ..tensors.tensor_conversion import TensorConversion, TensorConversionInfo
from .base_bids import BidsLikeDataset, BidsLikeDatasetConfig


def _read_tensor_conversion(
    serialized: Optional[dict],
) -> Optional[TensorConversionInfo]:
    """
    To read the field 'tensor_conversion' even when it is ``None``.
    """
    if serialized:
        return TensorConversionInfo.from_dict(serialized)
    return serialized


class BidsLikeTensorDatasetConfig(BidsLikeDatasetConfig):
    """Config class to check store info on tensor conversion."""

    tensor_conversion: Optional[TensorConversionInfo] = Field(
        default=None, reader=_read_tensor_conversion
    )


class BidsLikeTensorDataset(BidsLikeDataset):
    """
    Abstract class for :py:class:`~clinicadl.data.datasets.base.BidsLikeDataset` that works with tensors.
    """

    config: BidsLikeTensorDatasetConfig

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @property
    def converted(self) -> bool:
        """Whether tensor conversion was performed."""
        return self._tensor_conversion is not None

    @property
    def _tensor_conversion(self) -> Optional[TensorConversionInfo]:
        """Information on tensor conversion."""
        return self.config.tensor_conversion

    @_tensor_conversion.setter
    def _tensor_conversion(
        self, tensor_conversion: Optional[TensorConversionInfo]
    ) -> None:
        self.config.tensor_conversion = tensor_conversion
        if self._tensor_conversion:
            self._load_pt_masks()
            self._check_image_transforms()
        else:
            self._reset_conversion()

    @property
    def _initial_shape(self) -> Optional[tuple[int, int, int, int]]:
        if self.converted:
            return self._tensor_conversion.shape
        return None

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
        self._tensor_conversion = None
        converter = TensorConversion(self)
        converter.to_tensors(
            n_proc=n_proc,
            ignore_spacing=ignore_spacing,
            shape_warning=shape_warning,
            conversion_name=conversion_name,
            overwrite=overwrite,
            save_transforms=save_transforms,
            check_transforms=check_transforms,
        )
        self._tensor_conversion = converter.get_info()
        self._count_samples()

    def read_tensor_conversion(
        self,
        conversion_name: Optional[str] = None,
        check_transforms: bool = True,
        load_also: Optional[list[str]] = None,
    ) -> None:
        """
        To read an old tensor conversion.

        The function will check that the old conversion works with the current dataset`,
        i.e. that the images of the current (participant, session) pairs have been converted
        to tensors, as well as the potential masks.

        If transformed images have been saved, it will also check that the transforms
        applied before conversion match the image transforms of the current dataset,
        unless ``check_transforms=False``.

        See :py:meth:`~BidsLikeTensorDataset.to_tensors` for more information on
        conversion to tensors.

        Parameters
        ----------
        conversion_name : Optional[str], default=None
            The name of the tensor conversion to read. This is what you passed to :py:meth:`~BidsLikeTensorDataset.to_tensors`
            during conversion. If you passed ``None``, leave ``conversion_name`` to ``None``.
        check_transforms : bool, default=True
            Whether to check if the image transforms potentially applied before tensor conversion
            match the current ones. ``check_transforms=False`` is useful when you use custom transforms (i.e. transforms
            not in ``ClinicaDL``), which cannot be read by ``ClinicaDL`` and thus cannot be checked.

            .. note::
                If :py:meth:`~BidsLikeTensorDataset.to_tensors` was run with ``save_transforms=False``, no check will
                be performed as the tensors saved have not been transformed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match before setting ``check_transforms=False``.

        load_also : list[str], default=[]
            To load additional information potentially stored in ``.pt`` files. By default, only the image and the masks
            mentioned in the argument ``masks`` of the dataset will be loaded.

        Raises
        ------
        FileNotFoundError
            If there is no conversion named ``conversion_name``.
        TensorConversionError
            If the conversion mentioned doesn't work with the
            current dataset (not the same datatype, images not all converted, transforms
            mismatch, etc.).
        ValueError
            If an element of ``load_also`` was already passed in the arguments ``columns`` or ``masks``
            of the dataset.

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
                datatype=datatypes.PETLinear(
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
        :py:meth:`~BidsLikeTensorDataset.to_tensors`
        """
        if load_also:
            for name in load_also:
                if name in self.columns:
                    raise ValueError(
                        f"Cannot load the element '{name}', as you already pass this name in 'columns'."
                    )
                elif name in set(self.config._individual_mask_names).union(
                    self.config._common_mask_names
                ):
                    raise ValueError(
                        f"Cannot load the element '{name}', as you already pass this name in 'masks'."
                    )

        self._tensor_conversion = None
        converter = TensorConversion(self)
        converter.read_tensor_conversion(
            conversion_name=conversion_name,
            check_transforms=check_transforms,
            load_also=load_also,
        )
        self._tensor_conversion = converter.get_info()
        self._count_samples()

    def _check_conversion(self) -> None:
        """
        Checks if tensor conversion was performed.
        """
        if not self.converted:
            raise RuntimeError(
                "The operation you are attempting to perform requires your data to be converted into tensors. "
                "Please use 'to_tensors', or 'read_tensor_conversion' if the conversion has already been performed."
            )

    def _check_image_transforms(self) -> None:
        """
        Checks if transformations on the whole image were already performed.
        In this case, disable these transforms.
        """
        if self._tensor_conversion.transforms:
            self.transforms.image_transforms = tio.Compose([])

    def _load_pt_masks(self) -> None:
        """
        Converts raw masks to the associated tensor masks.
        """
        pt_masks = []
        for mask in self.common_masks:
            mask_pt_path = self._tensor_conversion.path_to_tensors(
                self._get_common_mask_path(mask.name)
            )
            pt_masks.append(Mask(mask_pt_path))

        self.common_masks = pt_masks

    def _reset_conversion(self) -> None:
        """
        Resets the attributes updated after the last conversion.
        """
        self.common_masks = list(map(self._read_mask, self.config._common_masks))
        self.transforms.image_transforms = self.config.transforms.image_transforms

    def _load_data(
        self, participant: str, session: str
    ) -> tuple[tio.Image, Path, dict[str, Any]]:
        self._check_conversion()

        data_path = self._tensor_conversion.path_to_tensors(
            self._get_image_path(participant, session)
        )
        data = torch.load(data_path, weights_only=True)

        to_keep = {}

        # image
        image = tio.ScalarImage(tensor=data[IMAGE], affine=data[AFFINE])
        del data[IMAGE]

        # individual masks
        for mask in self.individual_masks:
            to_keep[mask.name] = tio.LabelMap(
                tensor=data[mask.name], affine=data[AFFINE]
            )

        # load_also
        load_also = self._tensor_conversion.also
        for name in load_also:
            if load_also[name] == IMAGE:
                to_keep[name] = tio.ScalarImage(tensor=data[name], affine=data[AFFINE])
            elif load_also[name] == MASK:
                to_keep[name] = tio.LabelMap(tensor=data[name], affine=data[AFFINE])
            else:
                to_keep[name] = data[name]

        return image, data_path, to_keep

    @abstractmethod
    def _get_image_path(self, participant: str, session: str) -> Path:
        """
        Gets the path to the raw image.
        """

    @classmethod
    def _from_config(cls, config: BidsLikeTensorDatasetConfig) -> Self:
        dataset = cls(**config.to_raw_dict(exclude=[DF, TENSOR_CONVERSION]))
        dataset._df = config.df
        dataset._tensor_conversion = config.tensor_conversion

        return dataset
