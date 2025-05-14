from __future__ import annotations

import warnings
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import torch
import torchio as tio
from joblib import Parallel, delayed
from pydantic import SerializeAsAny, ValidationError, field_serializer
from tqdm import tqdm

from clinicadl.dictionary.suffixes import JSON
from clinicadl.dictionary.words import (
    AFFINE,
    IMAGE,
    LABEL,
    MASK,
    OTHER,
    PARTICIPANT,
    PREPROCESSING,
    SESSION,
    TRANSFORMS,
)
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import TransformConfig, get_transform_config
from clinicadl.transforms.types import Transform
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLTensorConversionError,
)
from clinicadl.utils.typing import PathType

from .datatypes.preprocessing import Preprocessing, get_preprocessing_config
from .structures import DataPoint, Mask

if TYPE_CHECKING:
    from .datasets import CapsDataset
logger = getLogger("clinicadl.data.tensor_conversion")


class TensorConversionInfo(ClinicaDLConfig):
    """
    To store relevant information on the conversion of
    Caps images to tensors.
    """

    preprocessing: SerializeAsAny[
        Preprocessing
    ]  # SerializeAsAny to have fields that are not in the base Preprocessing
    individual_masks: list[str]
    common_masks: list[str]
    also: dict[str, Literal[IMAGE, MASK, OTHER]]  # other information stored in .pt
    transforms: list[
        Union[str, Transform, TransformConfig]
    ]  # str: a description of the custom transform (see clinicadl.transforms.transforms.Transforms.serialize_transforms)
    spacing: Optional[tuple[float, float, float]]
    shape: Optional[tuple[int, int, int]]
    interrupted: bool = False
    participants_sessions: list[tuple[str, str]]

    @field_serializer("transforms")
    @classmethod
    def serialize_transforms(
        cls, transforms: list[Union[Transform, TransformConfig]]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of transforms that are not passed via
        TransformConfigs.
        """
        return Transforms.serialize_transforms(transforms)

    @classmethod
    def from_json(cls, json_path: Path) -> TensorConversionInfo:
        """
        Reads information on a conversion.
        """
        info = cls.read_json(json_path)

        try:
            if isinstance(info[TRANSFORMS], list):
                transforms = []
                for transform in info[TRANSFORMS]:
                    if isinstance(transform, dict):
                        transforms.append(get_transform_config(**transform))
                    else:  # a str describing the transform (see clinicadl.transforms.transforms.Transforms.serialize_transforms)
                        transforms.append(transform)
            else:
                raise ClinicaDLTensorConversionError(
                    f"{json_path} is not a valid tensor conversion file."
                    "Value for 'transforms' should be a list."
                )
            del info[TRANSFORMS]

            preprocessing = get_preprocessing_config(**info[PREPROCESSING])
            del info[PREPROCESSING]

            return TensorConversionInfo(
                preprocessing=preprocessing, transforms=transforms, **info
            )
        except ValidationError as exc:
            raise ClinicaDLTensorConversionError(
                f"{json_path} is not a valid tensor conversion file."
                "Some values have been corrupted and cannot be read."
            ) from exc


class TensorConversion:
    """
    To convert NIfTI files to tensors to speed up data loading during training or inference.

    Before conversion to tensors, transforms at the image level can be applied, in order not
    to have to compute them each time the image is loaded.
    Images are also (always) converted to the same coordinate system (RAS+).

    Parameters
    ----------
    caps_dataset : CapsDataset
        The CapsDataset on which conversion will be performed.
    """

    def __init__(self, caps_dataset: CapsDataset):
        self.caps_dataset = caps_dataset
        self.caps_reader = caps_dataset.caps_reader
        self.preprocessing = caps_dataset.preprocessing
        self.transform = caps_dataset.image_transform
        self.save_directory = self.caps_reader.tensor_conversion_json_dir

        self.json = None

        self._to_canonical = tio.ToCanonical()

        self._save_transforms = True
        self._ignore_spacing = False
        self._raise_warnings = True
        self._ref_image_spacing = None
        self._ref_image_shape = None
        self._output_shape = None
        self._output_spacing = None
        self._uniform_shape = True
        self._participants_sessions_converted = set()
        self._masks_converted = set()
        self._also = None

        self._currently_reading = None

    def get_info(self) -> TensorConversionInfo:
        """
        Gathers all relevant information on the conversion.

        Returns
        -------
        TensorConversionInfo
            a data structure that contains the information,
            with attributes 'preprocessing', 'participants_sessions',
            'individual_masks', 'common_masks', 'transforms', and 'spacing'.
        """
        individual_masks = set(
            [mask.name for mask in self.caps_dataset.individual_masks]
        )
        if isinstance(self.caps_dataset.label, Mask):
            individual_masks.add(self.caps_dataset.label.name)

        return TensorConversionInfo(
            preprocessing=self.preprocessing,
            participants_sessions=self._participants_sessions_converted,
            individual_masks=individual_masks,
            common_masks=self._masks_converted,
            also=self._also if self._also else {},
            transforms=self.caps_dataset.transforms.image_transforms
            if self._save_transforms
            else [],
            spacing=self._output_spacing,
            shape=self._output_shape,
        )

    def read_conversion(
        self,
        json_name: str,
        check_transforms: bool = True,
        load_also: Optional[list[str]] = None,
    ):
        """
        To read an old tensor conversion json and updates the states of the
        current TensorConversion object.

        Parameters
        ----------
        json_name : str
            the name of the json file (without `.json` suffix) in the folder 'tensor_extraction'
            of the caps directory.
        check_transforms : bool (optional, default=True)
            whether to checks if the image transforms potentially applied before tensor conversion
            match the current ones. Useful when you use custom transforms (i.e. transforms
            not in ClinicaDL), which cannot be checked.\n

            .. note::
                If ``to_tensors`` was run with ``save_transforms=False``, no check will
                be performed as the tensors saved have not been transformed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match.

        load_also : list[str] (optional, default=None)
            to load additional information potentially stored in `.pt` files. By default, only the image, the label, and masks
            mentioned in ``masks`` of the CapsDataset will be loaded.


        Raises
        ------
        FileNotFoundError
            if there is no json file named 'json_name' in the 'tensor_extraction' folder.
        ClinicaDLTensorConversionError
            if the json file is not a tensor conversion file produced by `convert_to_tensors`.
        ClinicaDLTensorConversionError
            if the conversion mentioned in the json file doesn't work with the
            current CapsDataset (not the same preprocessing, participants not
            all converted, etc.).
        """
        json_path = self._check_json_name(json_name)
        conversion_info = TensorConversionInfo.from_json(json_path)
        self._currently_reading = str(json_path)  # for potential error messages

        # do we talk about the same preprocessing?
        self._compare_preprocessing(conversion_info)

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
        self._participants_sessions_converted = set(
            self.caps_dataset.get_participant_session_couples()
        )
        self._masks_converted = set(
            mask.path.name for mask in self.caps_dataset.common_masks
        )
        self._output_spacing = conversion_info.spacing
        self._output_shape = conversion_info.shape
        self._save_transforms = transforms_saved
        self._also = {
            name: type_
            for name, type_ in conversion_info.also.items()
            if name in load_also
        }
        self.json = json_path

    def convert_to_tensors(
        self,
        json_name: PathType = "tensor_conversion",
        save_transforms: bool = True,
        n_proc: int = 1,
        ignore_spacing: bool = False,
        raise_warnings: bool = True,
        check_transforms: bool = True,
    ) -> None:
        """
        Performs conversion.

        Parameters
        ----------
        json_name : str (optional, default="tensor_conversion")
            the name of the json file where the information on the conversion
            (e.g. transforms applied) will be stored. The full path of
            the json file will be `{caps_directory}/prepare_data/tensor_conversion/{json_name}.json`.\n
            If the file already exists, TensorConversion will try to merge the old
            tensor conversion with the new one, if they concern the same type of data (e.g. same
            preprocessing, same transforms applied), otherwise an error will be raised.
        save_transforms : bool (optional, default=True)
            whether to save raw images as tensors (False) or images on which were applied image
            transforms (True).
        n_proc : int (optional, default=1)
            number of cores to use to parallelize the conversion.
        ignore_spacing : bool (optional, default=False)
            whether to ignore the check made on voxel spacings. If False, it will make sure that all
            images have the same voxel spacing.
            ..warning::In most medical image applications, all the images should have the same
            voxel spacing. Be sure that you don't care before disabling this check.
        raise_warnings : bool (optional, default=True)
            whether to raise different kinds of warnings detected during conversion (e.g. images with
            different shapes).
        check_transforms : bool (optional, default=True)
            If ``json_name`` already exists, TensorConversion will try to merge the old
            tensor conversion with the new one. ``check_transforms`` determines whether transforms
            should be checked during the merger. If ``True``, TensorConversion will check that
            the same transforms were applied during the two conversions.\n
            Useful when you use custom transforms (i.e. transforms not in ClinicaDL), which cannot be checked.\n

            .. note::
                If ``save_transforms=False``, no check will be performed as the tensors saved
                have not been transformed.

            .. warning::
                **To use carefully**. You must be sure that the transforms match.

        Raises
        ------
        ClinicaDLArgumentError
            if a json file with the same `json_name` already exists and the new conversion cannot
            be merged with the old one.
        ClinicaDLTensorConversionError
            if images don't have the same voxel spacing across (participant, session), and
            `ignore_spacing` is False.
        ClinicaDLTensorConversionError
            if some image-specific masks don't have the same shape and affine matrix as the image.

        Also raises a warning (only once) if images have different shapes across (participant, session)
        (unless `raise_warnings` is False).
        Also raises a warning if some `.pt` files already present in the caps directory will be
        overwritten (unless `raise_warnings` is False).
        """
        self._reset()
        self._save_transforms = save_transforms
        self._ignore_spacing = ignore_spacing
        self._raise_warnings = raise_warnings

        json_path = self._get_json_path(json_name, check_transforms=check_transforms)

        # process images and masks, and manage errors
        try:
            Parallel(n_jobs=n_proc, require="sharedmem")(
                delayed(self._transform_and_save_images)(participant, session)
                for participant, session in tqdm(
                    set(self.caps_dataset.get_participant_session_couples()).difference(
                        self._participants_sessions_converted
                    ),
                    desc="Converting images and image-specific masks",
                )
            )
            Parallel(n_jobs=n_proc, require="sharedmem")(
                delayed(self._transform_and_save_mask)(mask)
                for mask in tqdm(
                    [
                        mask
                        for mask in self.caps_dataset.common_masks
                        if mask.path.name not in self._masks_converted
                    ],
                    desc="Converting common masks",
                )
            )
        except Exception as exc:
            self._update_json_dir(json_path, interrupted=True)
            raise ClinicaDLTensorConversionError(
                "An error occurred during conversion. The images correctly converted before "
                f"the exception was raised can be found in {str(json_path)}. See exception traceback "
                "for more information."
            ) from exc

        # save info
        self._update_json_dir(json_path)
        self.json = json_path

    ### to process (participant, session) and masks individually ###
    def _transform_and_save_images(self, participant: str, session: str) -> None:
        """
        The processing function called for each (participant, session).
        It loads all the images associated to the (participant, session)
        (the image and the masks), applies image transforms, and saves them
        in a .pt file.
        """
        logger.debug("Conversion of (%s, %s).", participant, session)

        images = self._get_nifti_images(participant, session)
        images = self._transform(images)
        self._check_consistency_with_dataset(images)

        self._remove_common_mask(images)  # we don't want to save common mask
        pt_path = self.caps_reader.get_tensor_path(
            participant, session, self.preprocessing, check=False
        )
        self.save_images_as_tensors(images, pt_path)

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

        assert mask.path is not None  # TODO: check if a test is needed

        pt_path = self.caps_reader.path_to_tensor(mask.path)
        label_map = getattr(images, mask.name)
        self.save_mask_as_tensor(label_map, pt_path)

        self._masks_converted.add(mask.path.name)

    def _transform(self, images: DataPoint) -> DataPoint:
        """
        Puts all the images in RAS+ space and apply
        the transforms at the image level.
        """
        images = self._to_canonical(images)
        if self._save_transforms:
            return self.transform(images)
        else:
            return images

    ### to get the images ###
    def _get_nifti_images(self, participant: str, session: str) -> DataPoint:
        """
        Loads all the images associated to the (participant, session)
        (the image and the masks).

        Checks that all masks have the same shape as the image.
        Checks that all image-specific masks have the same
        affine matrix as the image.
        If `ignore_spacing` is not True, checks that all common
        masks have the same voxel spacing as the image.
        """
        image_path = self.caps_reader.get_image_path(
            participant, session, self.preprocessing
        )
        images = {IMAGE: image_path}

        # label
        label = self.caps_dataset.label
        if isinstance(label, Mask):
            images[LABEL] = label.get_associated_mask(image_path)
        else:
            images[LABEL] = None  # no use here if it is not an image

        # image-specific masks
        for mask in self.caps_dataset.individual_masks:
            images[mask.name] = mask.get_associated_mask(image_path)

        images = DataPoint(participant=participant, session=session, **images)
        self._check_affines_consistency(images)

        # common masks
        for mask in self.caps_dataset.common_masks:
            images.add_mask(mask.get_associated_mask(image_path), mask.name)

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
            raise ClinicaDLTensorConversionError(message) from exc

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
            raise ClinicaDLTensorConversionError(message) from exc

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
            raise ClinicaDLTensorConversionError(message) from exc

    ### to save tensors ###
    def save_images_as_tensors(self, images: DataPoint, path: PathType) -> None:
        """
        Saves all the images related to an image in the same .pt file.
        The affine matrix of the image is also saved in the file.

        More precisely, they are saved as a dict with at least the keys 'image'
        and 'affine'. Potential masks can be accessed via their name.

        Parameters
        ----------
        images : DataPoint
            a DataPoint containing the image and the associated masks.
        path : PathType
            where to save the images.
        """
        Path(path).parent.mkdir(exist_ok=True)

        images_dict = {}
        del images[PARTICIPANT]
        del images[SESSION]
        for name, value in images.items():
            if isinstance(value, tio.ScalarImage):
                images_dict[name] = value.tensor.float()
            elif isinstance(value, tio.LabelMap) and name == LABEL:
                images_dict[self.caps_dataset.label.name] = value.tensor.int()
            elif isinstance(value, tio.LabelMap) and name != LABEL:
                images_dict[name] = value.tensor.int()
            else:
                images_dict[name] = value

        images_dict[AFFINE] = torch.from_numpy(images.image.affine).float()

        path = Path(path)
        if path.is_file():
            logger.info("The file %s exists. It will be overwritten.", path)
        torch.save(images_dict, path)

    @staticmethod
    def save_mask_as_tensor(mask: tio.LabelMap, path: PathType) -> None:
        """
        Saves a common mask in a .pt file, along with its affine matrix.

        More precisely, it is saved as a dict with the keys 'mask' and
        'affine'.

        Parameters
        ----------
        mask : tio.LabelMap
            the mask, as a TorchIO LabelMap.
        path : PathType
            where to save the mask.
        """
        Path(path).parent.mkdir(exist_ok=True)

        mask_dict = {
            MASK: mask.tensor.int(),
            AFFINE: torch.from_numpy(mask.affine).float(),
        }

        path = Path(path)
        if path.is_file():
            logger.info("The file %s exists. It will be overwritten.", path)
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
        if not np.isclose(spacing, self._ref_image_spacing.spacing, rtol=1e-2).all():
            raise ClinicaDLTensorConversionError(
                "Different voxel spacings found in the CAPS dataset: "
                f"for example, voxel spacing is {spacing} in {image.path}, "
                f"but {tuple(float(s) for s in self._ref_image_spacing.spacing)} in {self._ref_image_spacing.path}.\n"
                "If you don't care about voxel spacing and want to ignore this error, set `ignore_spacing` "
                "to True."
            )

    def _check_shape(self, image: tio.Image) -> None:
        """
        Checks that the shape of an image is equal to the reference shape.
        """
        shape = image.spatial_shape
        if shape != self._ref_image_shape.spatial_shape:
            if (
                self._raise_warnings
                and self._uniform_shape  # to avoid raising to many warnings
            ):
                warnings.warn(
                    "Different image shapes found in the CAPS dataset: "
                    f"for example, {image.path} is {shape}, "
                    f"but {self._ref_image_shape.path} is {self._ref_image_shape.spatial_shape}.\n"
                    "It can be problematic if your network only accepts a specific shape.\n"
                    "If you don't want this warning to be raised, set `raise_warnings` "
                    "to False."
                )
            self._uniform_shape = False

    ### to save info in json ###
    def _update_json_dir(self, new_json: Path, interrupted: bool = False) -> None:
        """
        Saves the conversion information in a json file and updates
        the old json files (i.e. remove from them participant/session and
        masks that has been converted with the current conversion, because
        their .pt files have been overwritten).
        """
        self._compute_output_info()
        current_conversion = self.get_info()
        current_conversion.interrupted = interrupted
        try:
            current_conversion.write_json(new_json)
        except FileExistsError:  # resuming conversion
            current_conversion.update_json(new_json)
        self._update_old_jsons(current_conversion, new_json)

    def _update_old_jsons(
        self, current_conversion: TensorConversionInfo, new_json: Path
    ) -> None:
        """
        Iterates over all json files in 'tensor_conversion' and updates them.
        """
        for old_json_file in self.save_directory.iterdir():
            if old_json_file != new_json and old_json_file.suffix == JSON:
                old_conversion = TensorConversionInfo.from_json(old_json_file)

                if old_conversion.preprocessing == current_conversion.preprocessing:
                    self._update_participants_sessions(
                        old_conversion, current_conversion
                    )

                self._update_masks(old_conversion, current_conversion)

                old_conversion.update_json(old_json_file)

    def _update_participants_sessions(
        self,
        old_conversion: TensorConversionInfo,
        current_conversion: TensorConversionInfo,
    ) -> None:
        """
        Removes (participant, session) couples of the current conversion from
        an old conversion.
        """
        intersection = set(old_conversion.participants_sessions).intersection(
            set(current_conversion.participants_sessions)
        )

        if len(intersection) > 0:
            if self._raise_warnings:
                warning_message = "The following (participant, session) have already been converted:\n"
                for participant, session in intersection:
                    warning_message += f"   * ({participant}, {session})\n"
                warning_message += "The old tensors will be overwritten."
                warnings.warn(warning_message)

            old_conversion.participants_sessions = set(
                old_conversion.participants_sessions
            ).difference(set(current_conversion.participants_sessions))

    def _update_masks(
        self,
        old_conversion: TensorConversionInfo,
        current_conversion: TensorConversionInfo,
    ) -> None:
        """
        Removes common masks of the current conversion from an old conversion.
        """
        intersection = set(old_conversion.common_masks).intersection(
            set(current_conversion.common_masks)
        )

        if len(intersection) > 0:
            if self._raise_warnings:
                warning_message = "The following masks have already been converted:\n"
                for mask in intersection:
                    warning_message += f"   * {mask}\n"
                warning_message += "The old tensors will be overwritten."
                warnings.warn(warning_message)

            old_conversion.common_masks = set(old_conversion.common_masks).difference(
                set(current_conversion.common_masks)
            )

    def _check_json_name(self, json_name: str) -> Path:
        """
        Checks that a json named 'json_name' is indeed in 'tensor_extraction' folder.
        """
        json_path = (self.save_directory / json_name).with_suffix(JSON)
        if not json_path.is_file():
            raise FileNotFoundError(
                f"{json_path} does not exist, please give a valid 'json_name'."
            )
        return json_path

    def _get_json_path(
        self, json_name: PathType, check_transforms: bool = True
    ) -> Path:
        """
        Checks if 'json_name' exists. If it exists,
        tries to merge the two tensor conversions.
        """
        json_path = (self.save_directory / json_name).with_suffix(JSON)
        if json_path.is_file():
            try:
                self._merge_conversion(json_path, check_transforms=check_transforms)
            except ClinicaDLTensorConversionError as exc:
                raise ClinicaDLArgumentError(
                    f"{json_path} already exists, so ClinicaDL tried to merge the current tensor conversion "
                    "with the old one. But an error occurred, most likely because the two conversions concern "
                    "different kinds of data (e.g. different preprocessing, different transforms applied, different "
                    "masks used). See exception traceback for more details. If you want to run a new tensor conversion, "
                    "please give an available 'json_name'."
                ) from exc

        return json_path

    def _merge_conversion(
        self,
        json_path: Path,
        check_transforms: bool = True,
    ) -> None:
        """
        Tries to merge old conversion in `json_path` with the current one.
        Checks beforehand that they match.
        """
        conversion_info = TensorConversionInfo.from_json(json_path)
        self._currently_reading = str(json_path)  # for potential error messages

        # check that .pt files contain the same things
        self._compare_preprocessing(conversion_info)
        if self._save_transforms:
            if check_transforms:
                self._compare_transforms(conversion_info)
        else:
            if (
                conversion_info.transforms != []
            ):  # ensure no transform has been saved in old .pt files
                raise ClinicaDLTensorConversionError(
                    "'save_transforms' is set to False, but some transforms have already been saved "
                    f"in old tensor files associated to '{self._currently_reading}'."
                )
        self._compare_individual_masks(
            conversion_info, match_exactly=True
        )  # here, we want to have exactly the same masks in .pt files
        self._compare_also(conversion_info)

        # all checks passed, update current state
        if len(conversion_info.participants_sessions) > 0:
            ref_participant, ref_session = conversion_info.participants_sessions[0]
            ref_image = self._get_nifti_images(ref_participant, ref_session).image
            if conversion_info.spacing:
                self._ref_image_spacing = ref_image
            else:
                self._ignore_spacing = True
            if conversion_info.shape:
                self._ref_image_shape = ref_image
            else:
                self._uniform_shape = False

        self._participants_sessions_converted = set(
            conversion_info.participants_sessions
        )
        self._masks_converted = set(conversion_info.common_masks)
        self._also = conversion_info.also

    ### to see if a conversion works with the current CapsDataset ###
    def _compare_preprocessing(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that conversion has been applied on this preprocessing.
        """
        if old_conversion.preprocessing != self.preprocessing:
            raise ClinicaDLTensorConversionError(
                "The preprocessing mentioned in 'json_path' does not match the current "
                f"preprocessing. In '{self._currently_reading}', got {old_conversion.preprocessing}, "
                f"whereas current preprocessing is {self.preprocessing}"
            )

    def _compare_transforms(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that image transforms used during conversion match the current ones.
        """
        for transform in old_conversion.transforms:
            if not isinstance(transform, TransformConfig):
                raise ClinicaDLTensorConversionError(
                    f"Custom transforms have been used during the conversion associated to "
                    f"'{self._currently_reading}', e.g.: '{transform}'.\n"
                    "ClinicaDL cannot read such custom transforms. For ClinicaDL to be able "
                    "to read tensor conversion json files, use only transforms implemented in "
                    "ClinicaDL (see our documentation to know these transforms).\n"
                    "If you are sure that the transforms match, set 'check_transforms' to False."
                )

        caps_image_transforms = self.caps_dataset.transforms.image_transforms
        for transform in caps_image_transforms:
            if not isinstance(transform, TransformConfig):
                raise ClinicaDLTensorConversionError(
                    f"Custom transforms have been passed to CapsDataset, "
                    f"e.g.: '{transform}'.\n"
                    f"ClinicaDL cannot compare such custom transforms to those in '{self._currently_reading}'. "
                    "For ClinicaDL to be able to compare the current transforms to those used during "
                    "tensor conversion, use only transforms implemented in "
                    "ClinicaDL (see our documentation to know these transforms).\n"
                    "If you are sure that the transforms match, set 'check_transforms' to False."
                )

        if old_conversion.transforms != caps_image_transforms:
            raise ClinicaDLTensorConversionError(
                f"The image transforms found in '{self._currently_reading}' does not match those passed "
                f"in the CapsDataset. Got respectively '{old_conversion.transforms}'\n"
                f"and '{caps_image_transforms}'"
            )

    def _compare_individual_masks(
        self, old_conversion: TensorConversionInfo, match_exactly: bool = False
    ) -> None:
        """
        Checks that all individual masks have been converted.

        If `match_exactly`, it will check that the individual masks in .pt files
        match exactly the individual masks of the CapsDataset. Otherwise, it will
        only check that the .pt files have AT LEAST the individual masks required by the CapsDataset.
        """
        individual_masks_in_caps = {
            mask.name for mask in self.caps_dataset.individual_masks
        }
        if isinstance(self.caps_dataset.label, Mask):
            individual_masks_in_caps.add(self.caps_dataset.label.name)

        if match_exactly:
            sym_diff = individual_masks_in_caps.symmetric_difference(
                old_conversion.individual_masks
            )
            if len(sym_diff) > 0:
                raise ClinicaDLTensorConversionError(
                    f"There is a mismatch between image-specific masks in the current CapsDataset "
                    f"({individual_masks_in_caps}) and those converted in '{self._currently_reading}' "
                    f"({old_conversion.individual_masks})."
                )
        else:
            masks_not_converted = individual_masks_in_caps.difference(
                old_conversion.individual_masks
            )
            if len(masks_not_converted) > 0:
                raise ClinicaDLTensorConversionError(
                    "Some image-specific masks have not been converted by the conversion "
                    f"associated to '{self._currently_reading}': {masks_not_converted}"
                )

    def _compare_common_masks(self, old_conversion: TensorConversionInfo) -> None:
        """
        Checks that all common masks have been converted.
        """
        common_masks_in_caps = {
            mask.path.name for mask in self.caps_dataset.common_masks
        }
        masks_not_converted = common_masks_in_caps.difference(
            old_conversion.common_masks
        )
        if len(masks_not_converted) > 0:
            raise ClinicaDLTensorConversionError(
                f"Some masks have not been converted by the conversion associated to '{self._currently_reading}': "
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
                raise ClinicaDLTensorConversionError(
                    f"You passed '{info}' in 'load_also', but no such information was stored during "
                    f"the conversion associated to '{self._currently_reading}'"
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
            raise ClinicaDLTensorConversionError(
                f"There is a mismatch between the additional information in the CapsDataset "
                f"({list(current_also.keys())}) and that in '.pt' files saved with '{self._currently_reading}' "
                f"({list(old_conversion.also.keys())})."
            )
        for info in current_also:
            if current_also[info] != old_conversion.also[info]:
                raise ClinicaDLTensorConversionError(
                    "There is a mismatch between the additional information in the CapsDataset "
                    f"and that in '.pt' files saved with '{self._currently_reading}': "
                    f"'{info}' is of type '{current_also[info]}' in the current CapsDataset "
                    f"and of type '{old_conversion.also[info]}' in '.pt' files."
                )

    def _compare_participants_sessions(
        self, old_conversion: TensorConversionInfo
    ) -> None:
        """
        Checks that all (participant, session) have been converted.
        """
        caps_participants_session = set(
            self.caps_dataset.get_participant_session_couples()
        )
        not_converted = caps_participants_session.difference(
            old_conversion.participants_sessions
        )
        if len(not_converted) > 0:
            error_msg = (
                f"Some (participant, session) have not been converted with the conversion "
                f"mentioned in '{self._currently_reading}':\n"
            )
            for participant, session in not_converted:
                error_msg += f"   ({participant}, {session})\n"
            error_msg += (
                "\nUse `convert_to_tensors` method to relaunch a conversion on "
                "the whole dataset."
            )
            raise ClinicaDLTensorConversionError(error_msg)

    ### other utils ###
    def _reset(self) -> None:
        """
        Resets the state of the converter.
        """
        self.json = None

        self._save_transforms = True
        self._ignore_spacing = False
        self._raise_warnings = True
        self._ref_image_spacing = None
        self._ref_image_shape = None
        self._output_shape = None
        self._output_spacing = None
        self._uniform_shape = True
        self._participants_sessions_converted = set()
        self._masks_converted = set()
        self._also = None

        self._currently_reading = None

    def _remove_common_mask(self, images: DataPoint) -> None:
        """
        To remove common mask from a DataPoint, and keep only the
        mask specific to the (participant, session).
        """
        for mask in self.caps_dataset.common_masks:
            images.remove_image(mask.name)

    def _get_first_images(self) -> DataPoint:
        """
        To get an example of DataPoint.
        """
        participant, session = self.caps_dataset.get_participant_session_couples()[0]
        return self._get_nifti_images(participant, session)

    def _get_also(self, images: DataPoint) -> dict[str, Literal[IMAGE, MASK, OTHER]]:
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
                also_types[info] = IMAGE
            elif isinstance(images[info], tio.LabelMap):
                also_types[info] = MASK
            else:
                also_types[info] = OTHER

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
                self._output_shape = out_image.spatial_shape
