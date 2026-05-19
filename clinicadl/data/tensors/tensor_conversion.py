from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torchio as tio
from joblib import Parallel, delayed
from tqdm import tqdm

from clinicadl.io import Bids, TensorType
from clinicadl.io.bids.reader import DatasetDescription
from clinicadl.transforms.config import TransformConfig
from clinicadl.utils.dictionary.words import (
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.utils.exceptions import TensorConversionError
from clinicadl.utils.variables import BIDS_VERSION

from ..structures import DataPoint
from ..structures.images import TensorContent
from ..structures.sample import SAMPLE_FIELDS
from ..utils import DatasetChecker, SpatialCheck
from .utils import TensorDescription

if TYPE_CHECKING:
    from clinicadl.io import BidsFileType

    from ..datasets import BidsDataset

logger = getLogger(__name__)

DEFAULT_NAME = "raw"
DEFAULT_DESCRIPTION = "Raw images are converted without any transformation."


class TensorConversion:
    """
    To convert raw files to tensors to speed up data loading during training or inference.

    Before conversion to tensors, transforms at the image level can be applied in order not
    to have to compute them each time the image is loaded.
    Images are also converted to the same coordinate system (RAS+).

    Parameters
    ----------
    dataset : BidsDataset
        The :py:class:`clinicadl.data.datasets.BidsDataset` on which conversion will be performed.
    """

    def __init__(self, dataset: BidsDataset):
        self.dataset = dataset
        self.tensors_dir = self._init_tensors_dir()
        self.__save_transforms = False
        self._conversion_name: Optional[str] = None
        self._tensor_type: Optional[TensorType] = None
        self._tensor_description: Optional[TensorDescription] = None
        self._spatial_checkers: Optional[tuple[DatasetChecker, ...]] = None
        self._spacing_checker = DatasetChecker(
            [SpatialCheck.GLOBAL_SPACING]
        )  # will not raise warning but will just keep track of spacing
        self._shape_checker = DatasetChecker([SpatialCheck.GLOBAL_SHAPE])
        self._participants_sessions_converted = set()

    def _init_tensors_dir(self) -> Bids:
        tensors_path = self.dataset.image.bids.tensors_dir
        try:
            return Bids(tensors_path)
        except FileNotFoundError:
            DatasetDescription(
                Name="Conversions to tensors",
                BIDSVersion=BIDS_VERSION,
                DatasetType="derivative",
            ).to_json(tensors_path / Bids.DATASET_DESC_FILENAME)

        return Bids(tensors_path)

    def _reset(self) -> None:
        """
        Resets the state of the converter.
        """
        self.__save_transforms = False
        self._conversion_name = None
        self._tensor_type = None
        self._tensor_description = None
        self._spatial_checkers = None
        for checker in self._all_spatial_checkers:
            checker.reset()
        self._participants_sessions_converted = set()

    @property
    def _save_transforms(self) -> bool:
        """
        Whether to save transformed images or raw images.
        """
        return self.__save_transforms

    @_save_transforms.setter
    def _save_transforms(self, save_transforms: bool) -> None:
        if save_transforms and (
            len(self.dataset.transforms.image_transforms.transforms) == 0
        ):
            logger.info("save_transforms is True, but there are no image transform.")

        self.__save_transforms = save_transforms and (
            len(self.dataset.transforms.image_transforms.transforms) > 0
        )

    @property
    def conversion_name(self) -> Optional[str]:
        """
        Name of the current tensor conversion.
        """
        return self._conversion_name

    @conversion_name.setter
    def conversion_name(self, conversion_name: Optional[str]) -> None:
        if conversion_name:
            if conversion_name == DEFAULT_NAME:
                raise ValueError(
                    f"If you pass a conversion name, it cannot be '{DEFAULT_NAME}'."
                )
            self._conversion_name = conversion_name
        else:
            if self._save_transforms:
                raise ValueError(
                    "Please pass a conversion_name if save_transforms is True."
                )
            self._conversion_name = DEFAULT_NAME

    @property
    def tensor_type(self) -> TensorType:
        """
        ``TensorType`` output by the current conversion.
        """
        if not self._tensor_type:
            self._tensor_type = TensorType.from_source_file_types(
                conversion_name=self.conversion_name,
                image=self.dataset.image.file_type,
                individual_masks=[
                    mask.file_type for mask in self.dataset.individual_masks.values()
                ],
                common_masks=[mask.file for mask in self.dataset.common_masks.values()],
                transformed=self._save_transforms,
            )
        return self._tensor_type

    @property
    def tensor_description(self) -> TensorDescription:
        """
        A ``TensorDescription`` describing the current conversion.
        """
        if not self._tensor_description:
            self._tensor_description = TensorDescription(
                tensor_type=self.tensor_type,
                image=(self.dataset.image.bids.path, self.dataset.image.file_type),
                masks={
                    name: (mask.bids.path, mask.file_type)
                    for name, mask in self.dataset.individual_masks.items()
                }
                | {
                    name: mask.file.path
                    for name, mask in self.dataset.common_masks.items()
                },
                additional_data=[],
                transforms=self.dataset.transforms.config.image_transforms.to_raw()
                if self._save_transforms
                else [],
                spacing=None,
                spatial_shape=None,
                interrupted=True,
                description=DEFAULT_DESCRIPTION
                if self.conversion_name == DEFAULT_NAME
                else None,
                participants_sessions=pd.DataFrame(),
            )

        self._tensor_description.participants_sessions = (
            pd.DataFrame.from_records(
                list(self._participants_sessions_converted),
                columns=[PARTICIPANT_ID, SESSION_ID],
            )
            .sort_values([PARTICIPANT_ID, SESSION_ID])
            .reset_index(drop=True)
        )
        self._tensor_description.spacing = self.spacing
        self._tensor_description.spatial_shape = self.spatial_shape

        return self._tensor_description

    @property
    def spacing(self) -> Optional[tuple[float, float, float]]:
        """
        The voxel spacing in the current dataset. None if not consistent across the dataset
        or if no conversion performed yet.
        """
        if self._spacing_checker.enabled and (
            ref_sample := self._spacing_checker.ref_sample
        ):
            return ref_sample.spacing
        return None

    @property
    def spatial_shape(self) -> Optional[tuple[int, int, int]]:
        """
        The spatial shape of images in the current dataset. None if not consistent across the dataset
        or if no conversion performed yet.
        """
        if self._shape_checker.enabled and (
            ref_sample := self._shape_checker.ref_sample
        ):
            return ref_sample.spatial_shape
        return None

    @property
    def json(self) -> Path:
        """
        Path to the json file describing the current conversion.
        """
        return self.tensor_description.get_json_path(self.tensors_dir.path)

    @property
    def _all_spatial_checkers(self) -> Iterable[DatasetChecker]:
        """
        Returns all the spatial checkers instantiated.
        """
        default_checkers = (
            self._spacing_checker,
            self._shape_checker,
        )
        if not self._spatial_checkers:
            return default_checkers

        return self._spatial_checkers + default_checkers

    def to_tensors(
        self,
        conversion_name: Optional[str],
        spatial_checks: Optional[Iterable[str | SpatialCheck]],
        save_transforms: bool,
        description: Optional[str] = None,
        overwrite: bool = False,
        check_transforms: bool = True,
        n_proc: int = 1,
    ) -> TensorDescription:
        """
        Performs conversion to tensors.

        See :py:meth:`clinicadl.data.datasets.BidsDataset.to_tensors`.
        """
        self._reset()
        self._save_transforms = save_transforms
        self.conversion_name = conversion_name
        self._spatial_checkers = tuple(
            DatasetChecker(spatial_checks=[check]) for check in (spatial_checks or [])
        )

        if self.json.is_file() and overwrite:
            from ..utils import remove_tensors

            remove_tensors(self.json)
        elif self.json.is_file():
            self._merge_conversion_safely(check_transforms=check_transforms)

        if description:
            if not conversion_name:
                raise ValueError(
                    "You cannot pass a description if you don't pass a conversion_name."
                )
            self.tensor_description.description = description

        try:
            now = datetime.now().strftime("%H:%M:%S")
            Parallel(n_jobs=n_proc, require="sharedmem")(
                delayed(self._transform_and_save_images)(participant, session)
                for participant, session in tqdm(
                    self.dataset.get_participant_session_couples().difference(
                        self._participants_sessions_converted
                    ),
                    desc=f"{now} - Converting images and potential masks",
                )
            )
        except Exception as exc:
            raise TensorConversionError(
                "An error occurred during conversion. The images correctly converted before "
                f"the exception was raised can be found in {self.json}. See exception traceback "
                "for more information."
            ) from exc
        else:
            self.tensor_description.interrupted = False
            return self.tensor_description
        finally:
            self.tensor_description.write(self.tensors_dir.path)

    ### to process (participant, session) individually ###
    def _transform_and_save_images(self, participant: str, session: str) -> None:
        """
        The processing function called for each (participant, session).
        It loads all the images associated to the (participant, session)
        (the image and the masks), applies image transforms, and saves them
        in a .pt file.
        """
        logger.debug("Conversion of (%s, %s).", participant, session)

        pt_path = self.tensors_dir.build_path(
            self.tensor_type, participant=participant, session=session
        )

        images = self.dataset._get_images(participant, session)
        images = self._transform(images)
        self._spatial_check(images)
        self._save_images_as_tensors(images, pt_path)

        self._participants_sessions_converted.add((participant, session))

    def _transform(self, images: DataPoint) -> DataPoint:
        """
        Puts all the images in RAS+ space and applies
        the transforms at the image level.
        """
        images = tio.ToCanonical()(images)
        if self._save_transforms:
            return self.dataset.transforms.apply_image_transforms(images)
        return images

    def _spatial_check(self, images: DataPoint) -> None:
        """
        Performs spatial checks to check intra-sample and inter-sample
        consistency.
        """
        for checker in self._spatial_checkers:
            try:
                checker.check_data_point(images)
            except RuntimeError as e:
                logger.warning(e)
                checker.enabled = False

        for checker in (self._spacing_checker, self._shape_checker):
            try:
                checker.check_data_point(images)
            except RuntimeError:
                checker.enabled = False

    def _save_images_as_tensors(self, images: DataPoint, path: Path) -> None:
        """
        Saves all the images related to an image in the same .pt file.
        The affine matrices are also saved.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        for key in images.get_non_images_dict():
            if key in set(SAMPLE_FIELDS) - {PARTICIPANT, SESSION}:
                del images[key]

        content = TensorContent.from_datapoint(images)
        content.save(path)

        new_keys = set(content.images.keys()).union(content.masks.keys()).union(
            content.additional_data.keys()
        ) - {"image"}.union(self.tensor_description.masks.keys())
        self.tensor_description.additional_data = set(
            self.tensor_description.additional_data
        ).union(new_keys)

    ### merging conversions ###
    def _merge_conversion_safely(self, check_transforms: bool = True) -> None:
        """
        Tries to merge the current conversion with the old one.
        """
        try:
            self._merge_conversion(check_transforms=check_transforms)
        except TensorConversionError as exc:
            raise TensorConversionError(
                f"{self.json} already exists, so ClinicaDL tried to merge the current tensor conversion "
                "with the old one. But an error occurred, most likely because the two conversions concern "
                "different kinds of data (e.g., different file types, different transforms applied, different "
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
        old_conversion_info = TensorDescription.read(self.json)

        self._compare_images(old_conversion_info)
        self._compare_masks(old_conversion_info)
        if self._save_transforms:
            if check_transforms:
                self._compare_transforms(old_conversion_info)
        else:
            if old_conversion_info.transforms != []:
                raise TensorConversionError(
                    "'save_transforms' is set to False, but some transforms have already been saved "
                    "in the previous tensor files."
                )

        self._merge_spatial_checkers(old_conversion_info)
        self._participants_sessions_converted = set(
            zip(
                old_conversion_info.participants_sessions[PARTICIPANT_ID],
                old_conversion_info.participants_sessions[SESSION_ID],
            )
        )

    def _compare_images(self, old_conversion: TensorDescription) -> None:
        """
        Checks that the images match between two conversions.
        """
        _compare_subject_specific_images(
            old_conversion.image, self.tensor_description.image
        )

    def _compare_masks(self, old_conversion: TensorDescription) -> None:
        """
        Checks that the masks match between two conversions.
        """

        def _mask_description(mask: tuple | Path) -> str:
            if isinstance(mask, tuple):
                return "subject-specific mask"
            elif isinstance(mask, Path):
                return "common mask"

        if (old_masks := set(old_conversion.masks.keys())) != (
            new_masks := set(self.tensor_description.masks.keys())
        ):
            raise TensorConversionError(
                f"The masks in the previous dataset were: {old_masks}. The masks in the current one are: {new_masks}."
            )

        for name, old_mask in old_conversion.masks.items():
            new_mask = self.tensor_description.masks[name]

            if not isinstance(new_mask, type(old_mask)):
                raise TensorConversionError(
                    f"Previously, mask '{name}' was a {_mask_description(old_mask)}, currently it is a {_mask_description(new_mask)}."
                )

            if isinstance(old_mask, tuple):
                _compare_subject_specific_images(old_mask, new_mask, name=name)
            else:
                if str(old_mask) != str(new_mask):
                    raise TensorConversionError(
                        f"Previously, mask '{name}' was in {old_mask}, currently it is in {new_mask}."
                    )

    def _compare_transforms(self, old_conversion: TensorDescription) -> None:
        """
        Checks that image transforms used during conversion match the current ones.
        """

        def _error_msg(old: bool) -> str:
            return (
                f"Custom transforms {'have been' if old else 'are'} used in the {'previous' if old else 'current'} dataset, e.g.: '{transform}'. "
                "ClinicaDL cannot read such custom transforms. "
                "If you are sure that the transforms match, set 'check_transforms' to False."
            )

        for transform in old_conversion.transforms:
            if not isinstance(transform, TransformConfig):
                raise TensorConversionError(_error_msg(old=True))

        image_transforms = self.dataset.transforms.config.image_transforms.to_raw()
        for transform in image_transforms:
            if not isinstance(transform, TransformConfig):
                raise TensorConversionError(_error_msg(old=False))

        if old_conversion.transforms != image_transforms:
            raise TensorConversionError(
                f"The image transforms previously applied don't match with those passed to the current dataset. "
                f"Got respectively {old_conversion.transforms}\nand {image_transforms}"
            )

    def _merge_spatial_checkers(self, old_conversion_info: TensorDescription) -> None:
        """
        Resets the spatial checkers with values obtained in the previous dataset.
        """
        if not (
            (old_conversion_info.spacing or old_conversion_info.spatial_shape)
            and len(old_conversion_info.participants_sessions) > 0
        ):
            return

        first_row = old_conversion_info.participants_sessions.iloc[0]
        ref_participant, ref_session = first_row[PARTICIPANT_ID], first_row[SESSION_ID]
        spatial_shape = old_conversion_info.spatial_shape or (1, 1, 1)
        spacing = list(old_conversion_info.spacing or (1.0, 1.0, 1.0))
        affine = np.diag(spacing + [1.0])
        ref_sample = DataPoint(
            participant=ref_participant,
            session=ref_session,
            image=tio.ScalarImage(
                tensor=torch.zeros((1, *spatial_shape)),
                affine=affine,
            ),
        )

        for checker in self._all_spatial_checkers:
            if (
                checker.spatial_checks
                and SpatialCheck.GLOBAL_SPACING in checker.spatial_checks
            ):
                _update_checker(
                    checker,
                    ref_sample if old_conversion_info.spacing else None,
                )
            if (
                checker.spatial_checks
                and SpatialCheck.GLOBAL_SHAPE in checker.spatial_checks
            ):
                _update_checker(
                    checker,
                    ref_sample if old_conversion_info.spatial_shape else None,
                )


def _update_checker(checker: DatasetChecker, ref_sample: Optional[DataPoint]) -> None:
    """
    Updates the reference sample of a spatial checker, or disable the checker
    if no reference is provided.
    """
    if ref_sample:
        checker.ref_sample = ref_sample
    else:
        checker.enabled = False


def _compare_subject_specific_images(
    old: tuple[Path, BidsFileType],
    new: tuple[Path, BidsFileType],
    name: Optional[str] = None,
) -> None:
    """
    To compare images or subject-specific masks.
    """
    add_str = f"the mask '{name}' in " if name else ""
    if old[1] != new[1]:
        raise TensorConversionError(
            f"The file type of {add_str}the previous dataset does not match the current "
            f"file type. Previously, got {old[1]},"
            f"whereas current file type is {new[1]}"
        )
    if str(old[0]) != str(new[0]):
        raise TensorConversionError(
            f"The path to the BIDS of {add_str}the previous dataset does not match the current "
            f"path. Previously, got {old[0]}, "
            f"whereas current path is {new[0]}"
        )
