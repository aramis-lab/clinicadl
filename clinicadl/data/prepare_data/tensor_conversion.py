import warnings
from logging import getLogger
from typing import Optional, Union

import numpy as np
import torch
import torchio as tio

from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.types import Transform
from clinicadl.utils.exceptions import ClinicaDLCAPSError
from clinicadl.utils.typing import PathType

from ..datasets import CapsDataset
from ..structures import DataPoint
from .base import CapsProcessor, InfoJSON

logger = getLogger("clinicadl.prepare_data")


SAVE_SUB_DIRECTORY = "tensor_conversion"


class TensorConversionInfo(InfoJSON):
    """
    To store relevant information on the conversion of
    Caps images to tensors.
    """

    transforms: list[Union[Transform, TransformConfig]]
    ignore_spacing: bool
    spacing: Optional[tuple[float, float, float]]


class TensorConversion(CapsProcessor):
    """
    Converts NIfTI files to tensors to speed up data loading during training or inference.

    Before conversion to tensors, transforms at the image level are applied, in order not
    to have to compute them each time the image is loaded.
    Images are also converted to the same coordinate system (RAS+).

    Parameters
    ----------
    caps_dataset : CapsDataset
        The CapsDataset on which conversion will be performed.
    """

    def __init__(self, caps_dataset: CapsDataset):
        super().__init__(caps_dataset)
        self.save_directory = self.save_directory / SAVE_SUB_DIRECTORY
        self.transform = caps_dataset.image_transform
        self.ignore_spacing = False
        self._to_canonical = tio.ToCanonical()

    def convert_to_tensors(
        self,
        json_name: PathType = "tensor_conversion",
        n_proc: int = 1,
        ignore_spacing: bool = False,
    ) -> None:
        """
        Performs conversion.

        Parameters
        ----------
        json_name : PathType (optional, default="tensor_conversion")
            the name of the json file where the information on the conversion
            (e.g. transforms applied) will be stored. The full path of
            the json file will be `{caps_directory}/prepare_data/tensor_conversion/{json_name}.json`.
        n_proc : int (optional, default=1)
            number of cores to use to parallelize the conversion.
        ignore_spacing : bool (optional, default=False)
            whether to ignore the check made on voxel spacings. If False, it will make sure that all
            images have the same voxel spacing.
            ..warning::In most medical image applications, all the images should have the same
            voxel spacing. Be sure that you don't care before disabling this check. Otherwise, you
            can use the 'resample' method of 'CapsDataset' to uniformize voxel spacing.

        Raises
        ------
        FileExistsError
            if a json file with the same `json_name` already exists.
        ClinicaDLCAPSError
            if all images don't have the same voxel spacing, and `ignore_spacing`
            is False.
        ClinicaDLCAPSError
            if some images associated to the same (participant, session) don't have the same
            shape.

        Also raises a warning (only once) if some images in the Caps directory have different
        shapes.
        """
        if self.caps_dataset.tensor_conversion is not None:
            print(
                f"Images already converted to tensors (you passed {self.caps_dataset.tensor_conversion} "
                "for 'tensor_conversion' in CapsDataset)."
            )
            return None
        self.ignore_spacing = ignore_spacing
        self._process_caps(n_proc, json_name)

    @property
    def _store_info(self) -> type[TensorConversionInfo]:
        """
        Defines the data structure where to save the information
        on the conversion.
        """
        return TensorConversionInfo

    @property
    def _past_participle(self) -> str:
        """
        Past participle corresponding to the processing operation.
        Useful to write warnings or logs.
        """
        return "converted"

    def _reset(self) -> None:
        """
        Resets the state of the converter.
        """
        self._current_spacing = None
        self._current_shape = None
        self._current_image = None
        self._shape_warning_raised = False

    def _gather_info(self) -> TensorConversionInfo:
        """
        Gathers all relevant information on the conversion.
        """
        return TensorConversionInfo(
            preprocessing=self.preprocessing,
            participants_sessions=self.caps_dataset.get_participant_session_couples(),
            transforms=self.caps_dataset.transforms.image_transforms,
            ignore_spacing=self.ignore_spacing,
            spacing=self._current_spacing,
        )

    def _process(
        self, data: Union[tio.Image, DataPoint]
    ) -> Union[tio.Image, DataPoint]:
        """
        Converts images to the canonical space (RAS+) and applies
        image transforms passed by the user (i.e. the transforms
        that applies to the whole image).
        Accepts a single image or a collection of images related
        to the same (participant, session).

        Raises
        ------
        ClinicaDLCAPSError
            if all images don't have the same voxel spacing, and `ignore_spacing`
            is False.
        ClinicaDLCAPSError
            if some images associated to the same (participant, session) don't have the same
            shape.

        Also raises a warning (only once) if some images in the Caps directory have different
        shapes.
        """
        if isinstance(data, tio.Image):
            self._check_image(data)
        elif isinstance(data, DataPoint):
            self._check_shapes_consistency(data)
            for image in data.get_images(intensity_only=False):
                self._check_image(image)

        return self.transform(self._to_canonical(data))

    def _save_image(self, image: tio.Image) -> None:
        """
        Saves a processed image as a torch Tensor.
        """
        pt_path = self.caps_reader.path_to_tensor(image.path)
        if pt_path.is_file():
            logger.info("The file %s exists. It will be overwritten.", pt_path)
        if isinstance(image, tio.ScalarImage):
            tensor = image.tensor.float()
        elif isinstance(image, tio.LabelMap):
            tensor = image.tensor.int()
        torch.save(tensor, pt_path)  # pylint: disable=possibly-used-before-assignment

    def _update_caps_dataset(self, info: TensorConversionInfo) -> None:
        """
        Updates the state of the Caps Dataset.
        """
        self.caps_dataset.tensor_conversion = info

    def _check_shapes_consistency(self, images: DataPoint) -> None:
        """
        Checks if all images related to the same (participant, session)
        (i.e. the image and the associated masks) have the same shape.
        """
        try:
            images.spatial_shape
        except RuntimeError as exc:
            message = f"Inconsistent shapes were found for ({images.participant}, {images.session}):\n"
            for image in images.get_images(intensity_only=False):
                message += f"   {image.path}: {image.spatial_shape}\n"
            message += "\nThe masks associated to an image must have the same shape!"
            raise ClinicaDLCAPSError(message) from exc

    def _check_image(self, image: tio.Image) -> None:
        """
        Checks the voxel spacing and the shape of an image.
        """
        self._set_current_info(image)
        if not self.ignore_spacing:
            self._check_spacing(image=image)
        if not self._shape_warning_raised:  # to avoid raising to many warnings
            self._check_shape(image)

    def _set_current_info(self, image: tio.Image) -> None:
        """
        Sets the reference information to that of the first image
        seen.
        """
        if self._current_image is None:
            self._current_image = image.path
            self._current_shape = image.spatial_shape
            self._current_spacing = image.spacing

    def _check_spacing(self, image: tio.Image) -> None:
        """
        Checks that the voxel spacing of an image is (approximately)
        equal to the reference spacing.
        """
        spacing = tuple(float(s) for s in image.spacing)
        if not np.isclose(spacing, self._current_spacing, rtol=1e-2).all():
            raise ClinicaDLCAPSError(
                "Different voxel spacings found in the CAPS dataset: "
                f"for example, voxel spacing is {spacing} in {image.path}, "
                f"but {self._current_spacing} in {self._current_image}.\n"
                "Consider using the 'resample' method of 'CapsDataset' before "
                "converting images to tensors."
            )

    def _check_shape(self, image: tio.Image) -> None:
        """
        Checks that the shape of an image is equal to the reference shape.
        """
        shape = image.spatial_shape
        if shape != self._current_shape:
            warnings.warn(
                "Different image shapes found in the CAPS dataset: "
                f"for example, {image.path} is {shape}, "
                f"but {self._current_image} is {self._current_shape}.\n"
                "It can be problematic if your network only accepts a specific shape.\n"
                "If you want all the images to have the same shape, consider using "
                "'crop', 'pad', or 'crop_or_pad' methods of 'CapsDataset' before "
                "converting images to tensors."
            )
            self._shape_warning_raised = True
