from __future__ import annotations

import warnings
from datetime import datetime
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torchio as tio
from joblib import Parallel, delayed
from pydantic import (
    Field,
    PositiveFloat,
    PositiveInt,
)
from tqdm import tqdm
from typing_extensions import Self

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    AFFINE,
    IMAGE,
    LABEL,
    MASK,
    PARTICIPANT,
    SESSION,
    TENSOR_CONVERSION,
    TENSORS,
)
from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.factory import get_transform_from_dict
from clinicadl.transforms.types import Transform
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import (
    CannotReadJsonFieldError,
    TensorConversionError,
)
from clinicadl.utils.variables import SPACING_RTOL

from ..datatypes import DataType
from ..datatypes.factory import get_datatype_from_dict
from ..structures import DataPoint, Mask
from .utils import path_to_tensors

if TYPE_CHECKING:
    from ..datasets.tensor import TensorDataset

logger = getLogger("clinicadl.data.tensors.tensor_conversion")


class AlsoType(str, Enum):
    """Possible types for additional information stored in '.pt' files."""

    IMAGE = "image"
    MASK = "mask"
    OTHER = "other"


def _read_transform(
    serialized: Union[str, dict[str, Any]],
) -> Union[str, TransformConfig]:
    """
    Handles deserialization of transforms.
    """
    if isinstance(serialized, dict):
        return get_transform_from_dict(serialized)
    return serialized


class TensorConversionInfo(ClinicaDLConfig):
    """
    To store relevant information on the conversion of
    images to tensors.
    """

    conversion_name: str
    datatype: DataType = Field(reader=get_datatype_from_dict)
    individual_masks: list[str]
    also: dict[str, AlsoType]  # other information stored in .pt
    common_masks: list[str]
    transforms: list[
        Union[str, Transform, TransformConfig]
    ] = Field(  # str: to be able to read description of transform not passed via config classes
        reader=lambda x: list(map(_read_transform, x))
    )
    spacing: Optional[tuple[PositiveFloat, PositiveFloat, PositiveFloat]]
    shape: Optional[tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]]
    interrupted: bool
    participants_sessions: list[tuple[str, str]]

    @property
    def tensors_location(self) -> Path:
        """Folder where the tensors are stored."""
        return Path(TENSORS) / self.conversion_name

    @classmethod
    def from_json(cls, json_path: Path) -> Self:
        try:
            return super().from_json(json_path)
        except CannotReadJsonFieldError as e:
            raise CannotReadJsonFieldError(
                error=e.error, json_path=json_path, mention_kwargs=False
            )

    def path_to_tensors(self, path: Path) -> Path:
        """
        Given a path, returns the location of the associated ``.pt`` file.
        """
        return path_to_tensors(path, tensors_location=self.tensors_location)


class TensorConversion:
    """
    To convert raw files to tensors to speed up data loading during training or inference.

    Before conversion to tensors, transforms at the image level can be applied, in order not
    to have to compute them each time the image is loaded.
    Images are also converted to the same coordinate system (RAS+).

    Parameters
    ----------
    dataset : TensorDataset
        The :py:class:`clinicadl.data.datasets.tensor_dataset.TensorDataset` on which conversion will be performed.
    """

    def __init__(self, dataset: TensorDataset):
        self.dataset = dataset

        self.completed = False

        self._json = None
        self._conversion_name = None

        self._to_canonical = tio.ToCanonical()

        self._save_transforms = True
        self._ignore_spacing = False
        self._shape_warning = True
        self._ref_image_spacing = None
        self._ref_image_shape = None
        self._output_shape = None
        self._output_spacing = None
        self._uniform_shape = True
        self._participants_sessions_converted = set()
        self._masks_converted = set()
        self._also = None

    @property
    def conversion_name(self) -> Optional[str]:
        """
        Name of the current tensor conversion.
        """
        return self._conversion_name

    @conversion_name.setter
    def conversion_name(self, conversion_name: Optional[str]) -> None:
        if conversion_name:
            self._conversion_name = conversion_name
        else:
            self._conversion_name = Path(
                self.dataset.config.datatype.json_filename
            ).stem

    @property
    def tensors_path(self) -> Optional[Path]:
        """The folder where are saved the tensors (``sub-*/ses-*/{datatype}/{tensors_path}``)."""
        if self.conversion_name:
            return Path(TENSORS) / self.conversion_name
        return None

    @property
    def json(self) -> Optional[Path]:
        """The path to the json where the information is stored."""
        if self.conversion_name:
            return (
                self.dataset.config.directory / TENSOR_CONVERSION / self.conversion_name
            ).with_suffix(JSON)
        return None

    def get_info(self) -> TensorConversionInfo:
        """
        Gathers all relevant information on the conversion.

        Returns
        -------
        TensorConversionInfo
            A data structure that contains the information on the tensor conversion:
            - ``conversion_name``: the conversion name;
            - ``datatype``: the datatype of the data;
            - ``participants_sessions``: (participant, session) pairs converted;
            - ``individual_masks``: the individual masks present in the ``.pt`` files with the image;
            - ``also``: other information present in the ``.pt`` files in addition to the images and the masks;
            - ``common_masks``: the common masks converted;
            - ``transforms``: the image transforms applied before saving tensors;
            - ``spacing``: the spacing of the images (``None`` if not uniform across the images);
            - ``shape``: the shape of the images (``None`` if not uniform across the images);
            - ``interrupted``: whether the conversion has been interrupted by some errors.
        """
        individual_masks = set([mask.name for mask in self.dataset.individual_masks])

        return TensorConversionInfo(
            conversion_name=self.conversion_name,
            datatype=self.dataset.config.datatype,
            participants_sessions=self._participants_sessions_converted,
            individual_masks=individual_masks,
            also=self._also if self._also else {},
            common_masks=self._masks_converted,
            transforms=self.dataset.config.transforms.config.image_transforms.to_raw()
            if self._save_transforms
            else [],
            spacing=self._output_spacing,
            shape=self._output_shape,
            interrupted=not self.completed,
        )

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
        Performs conversion.

        See :py:meth:`clinicadl.data.datasets.tensor_dataset.TensorDataset.to_tensors`.
        """
        self._check_conversion_name(conversion_name, save_transforms)
        self._reset()
        self._save_transforms = save_transforms
        self._ignore_spacing = ignore_spacing
        self._shape_warning = shape_warning

        self.conversion_name = conversion_name
        if self.json.is_file() and overwrite:
            from ..utils import remove_tensors

            remove_tensors(self.json)
        elif self.json.is_file():
            self._merge_with_old_conversion(check_transforms=check_transforms)

        # process images and masks, and manage errors
        try:
            now = datetime.now().strftime("%H:%M:%S")
            Parallel(n_jobs=n_proc, require="sharedmem")(
                delayed(self._transform_and_save_images)(participant, session)
                for participant, session in tqdm(
                    self.dataset.get_participant_session_couples().difference(
                        self._participants_sessions_converted
                    ),
                    desc=f"{now} - Converting images and potential image-specific masks",
                )
            )
            Parallel(n_jobs=n_proc, require="sharedmem")(
                delayed(self._transform_and_save_mask)(mask)
                for mask in tqdm(
                    [
                        mask
                        for mask in self.dataset.common_masks
                        if mask.path.name not in self._masks_converted
                    ],
                    desc=f"{now} - Converting common masks",
                )
            )
        except Exception as exc:
            self._save_json()
            raise TensorConversionError(
                "An error occurred during conversion. The images correctly converted before "
                f"the exception was raised can be found in {str(self.json)}. See exception traceback "
                "for more information."
            ) from exc

        # save info
        self.completed = True
        self._save_json()

    def read_tensor_conversion(
        self,
        conversion_name: Optional[str] = None,
        check_transforms: bool = True,
        load_also: Optional[list[str]] = None,
        check_pt_files: bool = True,
    ):
        """
        To read an old tensor conversion json and updates the states of the
        current TensorConversion object.

        See :py:meth:`clinicadl.data.datasets.tensor_dataset.TensorDataset.read_tensor_conversion`.
        """
        self._reset()
        self.conversion_name = conversion_name
        self._check_conversion_exists()
        conversion_info = TensorConversionInfo.from_json(self.json)

        # do we talk about the same datatype?
        self._compare_datatype(conversion_info)

        # are the transforms applied during conversion the same?
        transforms_saved = conversion_info.transforms != []
        if transforms_saved and check_transforms:
            self._compare_transforms(conversion_info)

        # do we have the individual masks in the .pt files?
        self._compare_individual_masks(conversion_info)

        # have all the current common masks been converted?
        self._compare_common_masks(conversion_info)

        # do we have the information in 'load_also'?
        load_also = self._check_load_also(conversion_info, load_also)

        # are all (participant, session)s converted?
        self._compare_participants_sessions(conversion_info)

        # all checks passed, update current state
        self._participants_sessions_converted = (
            self.dataset.get_participant_session_couples()
        )
        self._masks_converted = set(
            mask.path.name for mask in self.dataset.common_masks
        )
        self._output_spacing = conversion_info.spacing
        self._output_shape = conversion_info.shape
        self._save_transforms = transforms_saved
        self._also = {
            name: type_
            for name, type_ in conversion_info.also.items()
            if name in load_also
        }

        # finally, check that pt files have not been deleted
        if check_pt_files:
            self._check_pt_files()
        self.completed = True

    def _get_data_pt_path(self, participant: str, session: str) -> Path:
        """
        Gets the path to the tensors associated to the (participant, session).
        """
        return self._path_to_tensors(self.dataset._get_image_path(participant, session))

    def _get_common_mask_pt_path(self, mask_name: str) -> Path:
        """
        Gets the path to the mask saved as a tensor.
        """
        return self._path_to_tensors(self.dataset._get_common_mask_path(mask_name))

    def _path_to_tensors(self, path: Path) -> Path:
        """
        Given a path, returns the location of the associated ``.pt`` file.
        """
        return path_to_tensors(path, tensors_location=self.tensors_path)

    ### to process (participant, session) and masks individually ###
    def _transform_and_save_images(self, participant: str, session: str) -> None:
        """
        The processing function called for each (participant, session).
        It loads all the images associated to the (participant, session)
        (the image and the masks), applies image transforms, and saves them
        in a .pt file.
        """
        logger.debug("Conversion of (%s, %s).", participant, session)

        images = self._get_raw_images(participant, session)
        images = self._transform(images)
        self._check_consistency_with_dataset(images)

        self._remove_common_mask(images)  # we don't want to save common masks here
        pt_path = self._get_data_pt_path(participant, session)
        self._save_images_as_tensors(images, pt_path)

        self._participants_sessions_converted.add((participant, session))
        if not self._also:
            self._also = self._get_also(images)

    def _transform_and_save_mask(self, mask: Mask) -> None:
        """
        The processing function called for masks common to
        all participants.
        It loads all the mask, transforms it, and saves it in
        a .pt file.
        """
        logger.debug("Conversion of mask '%s'.", mask.name)

        images = self._get_first_images()
        images = self._transform(images)

        pt_path = self._get_common_mask_pt_path(mask.name)
        label_map = getattr(images, mask.name)
        self._save_mask_as_tensor(label_map, pt_path)

        self._masks_converted.add(mask.path.name)

    def _transform(self, images: DataPoint) -> DataPoint:
        """
        Puts all the images in RAS+ space and apply
        the transforms at the image level.
        """
        images = self._to_canonical(images)
        if self._save_transforms:
            return self.dataset.config.transforms.apply_image_transforms(images)
        else:
            return images

    ### to get the images ###
    def _get_raw_images(self, participant: str, session: str) -> DataPoint:
        """
        Loads all the images associated to the (participant, session)
        (the image and the masks).

        Checks that all masks have the same shape as the image.
        Checks that all image-specific masks have the same
        affine matrix as the image.
        If 'ignore_spacing' is not True, checks that all common
        masks have the same voxel spacing as the image.
        """
        image_path = self.dataset._get_image_path(participant, session)
        images = {IMAGE: image_path}

        # image-specific masks
        for mask in self.dataset.individual_masks:
            images[mask.name] = mask.get_associated_mask(image_path)

        images = DataPoint(participant=participant, session=session, **images)
        self._check_affines_consistency(images)

        # common masks
        for mask in self.dataset.common_masks:
            images.add_mask(mask.get_associated_mask(), mask.name)

        self._check_shapes_consistency(images)
        if not self._ignore_spacing:
            self._check_spacings_consistency(
                images
            )  # with common masks, we don't check the affine matrix but only spacing

        return images

    @staticmethod
    def _check_shapes_consistency(images: DataPoint) -> None:
        """
        Checks if all images related to the same (participant, session)
        (i.e. the image and the associated masks) have the same shape.
        """
        try:
            images.spatial_shape
        except RuntimeError as exc:
            message = f"Inconsistent shapes were found for ({images.participant}, {images.session}):\n"
            for image in images.get_images(intensity_only=False):
                message += f"   * {image.path}: {image.spatial_shape}\n"
            message += "The masks associated to an image must have the same shape!"
            raise TensorConversionError(message) from exc

    @staticmethod
    def _check_affines_consistency(images: DataPoint) -> None:
        """
        Checks if all images related to the same (participant, session)
        (i.e. the image and the associated masks) have the same affine matrix.
        """
        try:
            images.affine
        except RuntimeError as exc:
            message = f"Inconsistent affine matrices were found for ({images.participant}, {images.session}):\n"
            for image in images.get_images(intensity_only=False):
                message += f"   * {image.path}:\n {image.affine}\n"
            message += (
                "The masks associated to an image must have the same affine matrix!"
            )
            raise TensorConversionError(message) from exc

    @staticmethod
    def _check_spacings_consistency(images: DataPoint) -> None:
        """
        Checks if all images related to the same (participant, session)
        (i.e. the image and the associated masks) have the same voxel spacings.
        """
        try:
            images.spacing
        except RuntimeError as exc:
            message = f"Inconsistent voxel spacings were found for ({images.participant}, {images.session}):\n"
            for image in images.get_images(intensity_only=False):
                message += f"   {image.path}: {image.spacing}\n"
            message += (
                "For a mask to be used on an image, it must have the same spacing as the image!\n"
                "If you don't care about voxel spacing and want to ignore this error, set `ignore_spacing` "
                "to True."
            )
            raise TensorConversionError(message) from exc

    ### to save tensors ###
    def _save_images_as_tensors(self, images: DataPoint, path: Path) -> None:
        """
        Saves all the images related to an image in the same .pt file.
        The affine matrix of the image is also saved in the file.

        More precisely, they are saved as a dict with at least the keys 'image'
        and 'affine'. Potential masks can be accessed via their name.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        images_dict = {}
        del images[PARTICIPANT]
        del images[SESSION]
        del images[LABEL]
        for name, value in images.items():
            if isinstance(value, tio.ScalarImage):
                images_dict[name] = value.tensor.float()
            elif isinstance(value, tio.LabelMap) and name != LABEL:
                images_dict[name] = value.tensor.int()
            else:
                images_dict[name] = value

        images_dict[AFFINE] = torch.from_numpy(images.image.affine).float()

        torch.save(images_dict, path)

    @staticmethod
    def _save_mask_as_tensor(mask: tio.LabelMap, path: Path) -> None:
        """
        Saves a common mask in a .pt file, along with its affine matrix.

        More precisely, it is saved as a dict with the keys 'mask' and
        'affine'.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        mask_dict = {
            MASK: mask.tensor.int(),
            AFFINE: torch.from_numpy(mask.affine).float(),
        }

        torch.save(mask_dict, path)

    ### to check consistency across the dataset
    def _check_consistency_with_dataset(self, images: DataPoint) -> None:
        """
        Checks that the voxel spacing and the shape for a
        (participant, session) is consistent with the rest of
        the dataset.
        """
        image = images.image
        self._set_ref_info(image)
        if not self._ignore_spacing:
            self._check_spacing(image)
        self._check_shape(image)

    def _set_ref_info(self, image: tio.Image) -> None:
        """
        Sets the reference information to that of the first image
        seen.
        """
        if self._ref_image_spacing is None:
            self._ref_image_spacing = image
        if self._ref_image_shape is None:
            self._ref_image_shape = image

    def _check_spacing(self, image: tio.Image) -> None:
        """
        Checks that the voxel spacing of an image is (approximately)
        equal to the reference spacing.
        """
        spacing = tuple(float(s) for s in image.spacing)
        if not np.isclose(
            spacing, self._ref_image_spacing.spacing, rtol=SPACING_RTOL
        ).all():
            raise TensorConversionError(
                "Different voxel spacings found in the dataset: "
                f"for example, voxel spacing is {spacing} in {image.path}, "
                f"but {tuple(float(s) for s in self._ref_image_spacing.spacing)} in {self._ref_image_spacing.path}.\n"
                "If you don't care about voxel spacing and want to ignore this error, set `ignore_spacing` "
                "to True."
            )

    def _check_shape(self, image: tio.Image) -> None:
        """
        Checks that the shape of an image is equal to the reference shape.
        """
        shape = image.shape
        if shape != self._ref_image_shape.shape:
            if (
                self._shape_warning
                and self._uniform_shape  # to avoid raising to many warnings
            ):
                warnings.warn(
                    "Different image shapes found in the dataset: "
                    f"for example, {image.path} is {shape}, "
                    f"but {self._ref_image_shape.path} is {self._ref_image_shape.shape}.\n"
                    "It can be problematic if your network only accepts a specific shape.\n"
                    "If you don't want this warning to be raised, set `shape_warning` "
                    "to False."
                )
            self._uniform_shape = False

    ### to save info in json ###
    def _save_json(self) -> None:
        """
        Saves the conversion information in a json file, or updates
        the json file if conversions were merged.
        """
        self._compute_output_info()
        conversion_info = self.get_info()
        conversion_info.to_json(self.json, overwrite=True)

    def _check_conversion_exists(self) -> None:
        """
        Checks that there is a conversion associated to 'self.json'.
        """
        if not self.json.is_file():
            raise FileNotFoundError(
                f"{self.json} does not exist, please give a valid 'conversion_name'."
            )

    def _merge_with_old_conversion(self, check_transforms: bool = True) -> None:
        """
        Tries to merge the current conversion with the old one.
        """
        try:
            self._merge_conversion(check_transforms=check_transforms)
        except TensorConversionError as exc:
            raise TensorConversionError(
                f"{str(self.json)} already exists, so ClinicaDL tried to merge the current tensor conversion "
                "with the old one. But an error occurred, most likely because the two conversions concern "
                "different kinds of data (e.g. different datatype, different transforms applied, different "
                "masks used).\n"
                "See exception traceback for more details. If you want to run a new tensor conversion, "
                "please give an available 'conversion_name'."
            ) from exc

    def _merge_conversion(
        self,
        check_transforms: bool = True,
    ) -> None:
        """
        Tries to merge the old conversion with the current one.
        Checks beforehand that they match.
        """
        old_conversion_info = TensorConversionInfo.from_json(self.json)

        # check that .pt files contain the same things
        self._compare_datatype(old_conversion_info)
        if self._save_transforms:
            if check_transforms:
                self._compare_transforms(old_conversion_info)
        else:
            if (
                old_conversion_info.transforms != []
            ):  # ensure no transform has been saved in old .pt files
                raise TensorConversionError(
                    "'save_transforms' is set to False, but some transforms have already been saved "
                    f"in old tensor files associated to '{str(self.json)}'."
                )
        self._compare_individual_masks(
            old_conversion_info, match_exactly=True
        )  # here, we want to have exactly the same masks in .pt files
        self._compare_also(old_conversion_info)

        # all checks passed, update current state
        if len(old_conversion_info.participants_sessions) > 0:
            ref_participant, ref_session = old_conversion_info.participants_sessions[0]
            ref_image = self._get_raw_images(ref_participant, ref_session).image
            if old_conversion_info.spacing:
                self._ref_image_spacing = ref_image
            else:
                self._ignore_spacing = True
            if old_conversion_info.shape:
                self._ref_image_shape = ref_image
            else:
                self._uniform_shape = False

        self._participants_sessions_converted = set(
            old_conversion_info.participants_sessions
        )
        self._masks_converted = set(old_conversion_info.common_masks)
        self._also = old_conversion_info.also

    ### to see if a conversion works with the current dataset ###
    def _compare_datatype(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that conversion has been applied on this datatype.
        """
        if old_conversion.datatype != self.dataset.config.datatype:
            raise TensorConversionError(
                "The datatype of the old conversion does not match the current "
                f"datatype. Previously, got '{old_conversion.datatype}' (see '{str(self.json)}'),\n"
                f"whereas current datatype is '{self.dataset.config.datatype}'"
            )

    def _compare_transforms(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that image transforms used during conversion match the current ones.
        """
        for transform in old_conversion.transforms:
            if not isinstance(transform, TransformConfig):
                raise TensorConversionError(
                    f"Custom transforms have been used during the old conversion, e.g.: '{transform}', (see {str(self.json)}).\n"
                    "ClinicaDL cannot read such custom transforms. For ClinicaDL to be able "
                    "to read tensor conversion json files, use only transforms supported natively in "
                    "ClinicaDL (see our documentation to know these transforms).\n"
                    "If you are sure that the transforms match, set 'check_transforms' to False."
                )

        image_transforms = (
            self.dataset.config.transforms.config.image_transforms.to_raw()
        )
        for transform in image_transforms:
            if not isinstance(transform, TransformConfig):
                raise TensorConversionError(
                    f"Custom transforms have been passed to the dataset, e.g.: '{transform}'.\n"
                    f"ClinicaDL cannot compare such custom transforms to those applied during the old conversion (see '{str(self.json)}'). "
                    "For ClinicaDL to be able to compare the current transforms to those used during "
                    "the old tensor conversion, use only transforms supported natively in "
                    "ClinicaDL (see our documentation to know these transforms).\n"
                    "If you are sure that the transforms match, set 'check_transforms' to False."
                )

        if old_conversion.transforms != image_transforms:
            raise TensorConversionError(
                f"The image transforms applied during the old conversion (see '{str(self.json)}') "
                f"does not match those passed to the dataset. Got respectively '{old_conversion.transforms}'\n"
                f"and '{image_transforms}'"
            )

    def _compare_individual_masks(
        self, old_conversion: TensorConversionInfo, match_exactly: bool = False
    ) -> None:
        """
        Checks that all individual masks have been converted.

        If 'match_exactly', it will check that the individual masks in .pt files
        match exactly the individual masks of the dataset. Otherwise, it will
        only check that the .pt files have AT LEAST the individual masks required by the dataset.
        """
        individual_masks = {mask.name for mask in self.dataset.individual_masks}

        if match_exactly:
            sym_diff = individual_masks.symmetric_difference(
                old_conversion.individual_masks
            )
            if len(sym_diff) > 0:
                raise TensorConversionError(
                    f"There is a mismatch between image-specific masks in the current dataset "
                    f"({individual_masks}) and those already converted (see '{str(self.json)}'): "
                    f"({old_conversion.individual_masks})."
                )
        else:
            masks_not_converted = individual_masks.difference(
                old_conversion.individual_masks
            )
            if len(masks_not_converted) > 0:
                raise TensorConversionError(
                    f"Some image-specific masks have not been converted (see '{str(self.json)}'): "
                    f"{masks_not_converted}"
                )

    def _compare_common_masks(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that all common masks have been converted.
        """
        common_masks = {mask.path.name for mask in self.dataset.common_masks}
        masks_not_converted = common_masks.difference(old_conversion.common_masks)
        if len(masks_not_converted) > 0:
            raise TensorConversionError(
                f"Some masks have not been converted (see '{str(self.json)}'): "
                f"{masks_not_converted}"
            )

    def _check_load_also(
        self, old_conversion: TensorConversionInfo, also: Optional[list[str]]
    ) -> list[str]:
        """
        Checks that the information in 'load_also' is effectively in the `.pt` files.
        """
        also = [] if also is None else also

        for info in also:
            if info not in old_conversion.also:
                raise TensorConversionError(
                    f"You asked '{info}' in 'load_also', but no such information was stored during "
                    f"conversion (see '{str(self.json)}')."
                )

        return also

    def _compare_also(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that the additional information in 'old_conversion' is the same as the current
        additional information.
        """
        images = self._get_first_images()
        images = self._transform(images)
        self._remove_common_mask(images)
        current_also = self._get_also(images)
        sym_diff = set(current_also.keys()).symmetric_difference(
            set(old_conversion.also.keys())
        )
        if len(sym_diff) > 0:
            raise TensorConversionError(
                f"There is a mismatch between the additional information currently saved "
                f"({list(current_also.keys())}) and that in the '.pt' files "
                f"({list(old_conversion.also.keys())}). See details in 'also' section in '{str(self.json)}'."
            )
        for info, type_ in current_also.items():
            if type_ != old_conversion.also[info]:
                raise TensorConversionError(
                    "There is a mismatch between the additional information currently saved "
                    f"and that in the '.pt' files (see 'also' in '{str(self.json)}'):\n"
                    f"'{info}' is of type '{type_}' in the current dataset "
                    f"and of type '{old_conversion.also[info]}' in the '.pt' files."
                )

    def _compare_participants_sessions(
        self, old_conversion: TensorConversionInfo
    ) -> None:
        """
        Checks that all (participant, session) have been converted.
        """
        participants_session = self.dataset.get_participant_session_couples()
        not_converted = participants_session.difference(
            old_conversion.participants_sessions
        )
        if len(not_converted) > 0:
            error_msg = f"Some (participant, session) have not been converted (see '{str(self.json)}'):\n"
            for participant, session in not_converted:
                error_msg += f"   ({participant}, {session})\n"
            error_msg += (
                "\nUse `to_tensors` method to relaunch a conversion on "
                "the whole dataset."
            )
            raise TensorConversionError(error_msg)

    ### to check that all pt files exists ###
    def _check_pt_files(self) -> None:
        """
        Checks that all pt files exist.
        """
        pt_files: list[Path] = []
        for participant, session in self._participants_sessions_converted:
            pt_path = self._get_data_pt_path(participant, session)
            pt_files.append(pt_path)
        for mask_path in self._masks_converted:
            pt_path = self._get_common_mask_pt_path(Mask.get_mask_name(mask_path))
            pt_files.append(pt_path)

        for path in pt_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Tensor conversion was performed, as suggested by the presence of {str(self.json)}. "
                    f"Nevertheless, file {str(path)} cannot be found. The tensors have probably been deleted "
                    "after conversion. Please rerun 'to_tensors' to generate the tensor files again."
                )

    ### other utils ###
    @staticmethod
    def _check_conversion_name(
        conversion_name: Optional[str], save_transforms: bool
    ) -> None:
        """Checks if 'conversion_name' is valid."""
        if conversion_name and conversion_name.startswith("default"):
            raise ValueError("'conversion_name' can't start with default")
        elif conversion_name is None and save_transforms:
            raise ValueError(
                "If 'save_transforms' is True, 'conversion_name' cannot be None."
            )

    def _reset(self) -> None:
        """
        Resets the state of the converter.
        """
        self.completed = False

        self._json = None
        self._conversion_name = None

        self._save_transforms = True
        self._ignore_spacing = False
        self._shape_warning = True
        self._ref_image_spacing = None
        self._ref_image_shape = None
        self._output_shape = None
        self._output_spacing = None
        self._uniform_shape = True
        self._participants_sessions_converted = set()
        self._masks_converted = set()
        self._also = None

    def _remove_common_mask(self, images: DataPoint) -> None:
        """
        To remove common mask from a DataPoint, and keep only the
        mask specific to the (participant, session).
        """
        for mask in self.dataset.common_masks:
            images.remove_image(mask.name)

    def _get_first_images(self) -> DataPoint:
        """
        To get an example of DataPoint.
        """
        participant, session = self.dataset.get_participant_session_couples().pop()
        return self._get_raw_images(participant, session)

    def _get_also(self, images: DataPoint) -> dict[str, AlsoType]:
        """
        To get the list of additional keys in DataPoint, and their types.
        """
        also = (
            set(images.keys())
            .difference([IMAGE, AFFINE, PARTICIPANT, SESSION, LABEL])
            .difference(self.get_info().individual_masks)
        )
        also_types = {}
        for info in also:
            if isinstance(images[info], tio.ScalarImage):
                also_types[info] = AlsoType.IMAGE
            elif isinstance(images[info], tio.LabelMap):
                also_types[info] = AlsoType.MASK
            else:
                also_types[info] = AlsoType.OTHER

        return also_types

    def _compute_output_info(self) -> None:
        """
        Gets output spacing and shape.
        """
        if (not self._ignore_spacing) or self._uniform_shape:
            images = self._get_first_images()
            out_image = self._transform(images).image

            if not self._ignore_spacing:
                self._output_spacing = (
                    out_image.spacing
                )  # no error before so they all have the same spacing
            if self._uniform_shape:
                self._output_shape = out_image.shape
