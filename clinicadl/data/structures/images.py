from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, TypeVar

import torch
import torchio as tio
from typing_extensions import Self

from clinicadl.io.bids import Bids, BidsFileType
from clinicadl.utils.bids import BidsFile
from clinicadl.utils.dictionary.suffixes import JSON
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType

from .datapoint import DataPoint

ImageT = TypeVar("ImageT")

SOURCE_FILES_KEY = "Sources"
PATH_PREFIX = "file://"


class SubjectSpecificImage(Generic[ImageT]):
    image_type: ImageT

    def __init__(self, bids: Bids, file_type: BidsFileType):
        self.bids = bids
        self.file_type = file_type

    def get(self, participant: str, session: str) -> ImageT:
        """
        Loads the image associated to the (participant, session) pair.

        Parameters
        ----------
        participant : str
            The participant id (e.g., "sub-xxx").
        session : str
            The session id (e.g., "ses-xxx").

        Returns
        -------
        tio.Image
            The image in a :py:class:`torchio.Image`.
        """
        path = self.bids.get_path(
            self.file_type, participant=participant, session=session
        )

        return self.image_type(path=path)


class Image(SubjectSpecificImage[tio.ScalarImage]):
    """
    To handle image loading from a :term:`BIDS` given a :py:class:`~clinicadl.io.BidsFileType`.
    """

    image_type = tio.ScalarImage


class IndividualMask(SubjectSpecificImage[tio.LabelMap]):
    """
    To handle image-specific mask loading from a :term:`BIDS` given a :py:class:`~clinicadl.io.BidsFileType`.
    """

    image_type = tio.LabelMap


class CommonMask:
    """
    To handle non-image-specific mask loading.
    """

    def __init__(self, path: PathType):
        self.file = BidsFile(path)
        self._mask: Optional[tio.LabelMap] = None  # lazy loading

    def get(self) -> tio.LabelMap:
        """
        Returns
        -------
        tio.LabelMap
            The mask in a :py:class:`torchio.LabelMap`.
        """
        if not self._mask:
            self._mask = tio.LabelMap(path=self.file.path)

        return self._mask


@dataclass
class TensorContent:
    """
    The content of a ``.pt`` file saved by ``ClinicaDL`` during
    tensor conversion.
    """

    images: dict[str, tio.ScalarImage]
    masks: dict[str, tio.LabelMap]
    additional_data: dict[str, Any]
    paths: list[Path]

    @classmethod
    def from_datapoint(
        cls,
        data_point: DataPoint,
        include: Optional[Iterable[str]] = None,
    ) -> Self:
        """
        To convert a :py:class:`~clinicadl.data.structures.DataPoint` to a ``TensorContent``.

        Parameters
        ----------
        data_point : DataPoint
            The input ``DataPoint``.
        include : Optional[Iterable[str]], default=None
            To keep only certain keys of the ``DataPoint`` in the ``.pt`` file.

        Returns
        -------
        Self
            A ``TensorContent``.
        """
        return cls(
            images=data_point.get_images_dict(include=include),
            masks=data_point.get_masks_dict(include=include),
            additional_data=data_point.get_non_images_dict(
                include=include, exclude=["participant", "session"]
            ),
            paths=[
                image.path
                for image in data_point.get_images_dict(
                    include=include, intensity_only=False
                ).values()
                if image.path
            ],
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        To create a ``TensorContent`` from a ``.pt`` file.

        Parameters
        ----------
        path : Path
            The input file.

        Returns
        -------
        Self
            The content of the file in ``TensorContent``.
        """
        content = torch.load(path, weights_only=False)

        images = {
            name: tio.ScalarImage(tensor=mask[0], affine=mask[1])
            for name, mask in content["images"].items()
        }
        masks = {
            name: tio.LabelMap(tensor=mask[0], affine=mask[1])
            for name, mask in content["masks"].items()
        }

        paths: list[str] = read_json(path.with_suffix(JSON))[SOURCE_FILES_KEY]

        return cls(
            images=images,
            masks=masks,
            additional_data=content["additional_data"],
            paths=[p.replace(PATH_PREFIX, "") for p in paths],
        )

    def save(self, path: Path) -> None:
        """
        To save the current ``TensorContent`` in a ``.pt`` file.

        Parameters
        ----------
        path : Path
            The path of the file.
        """
        to_save = dict()
        to_save["images"] = {
            name: (image.tensor, image.affine) for name, image in self.images.items()
        }
        to_save["masks"] = {
            name: (mask.tensor, mask.affine) for name, mask in self.masks.items()
        }
        to_save["additional_data"] = self.additional_data

        torch.save(to_save, path)
        write_json(
            path.with_suffix(JSON),
            {SOURCE_FILES_KEY: [PATH_PREFIX + str(p) for p in self.paths]},
        )


class Tensor(SubjectSpecificImage[DataPoint]):
    """
    To handle tensor loading from a :term:`BIDS` given a :py:class:`~clinicadl.io.BidsFileType`.
    The fields to keep from the tensor files may be specified.
    """

    def __init__(
        self,
        bids: Bids,
        file_type: BidsFileType,
        to_load: Optional[Iterable[str]] = None,
    ):
        super().__init__(bids, file_type)
        self.to_load = to_load

    def get(self, participant: str, session: str) -> DataPoint:
        path = self.bids.get_path(
            self.file_type, participant=participant, session=session
        )

        tensors = TensorContent.load(path)

        if self.to_load is not None:
            to_load = set(self.to_load).union({"image"})
        else:
            to_load = None

        to_keep = (
            _filter_dict(tensors.images, to_load)
            | _filter_dict(tensors.masks, to_load)
            | _filter_dict(tensors.additional_data, to_load)
        )

        return DataPoint(
            **to_keep,
            participant=participant,
            session=session,
            image_path=path,
            file_type=self.file_type,
        )


T = TypeVar("T")


def _filter_dict(
    dict_: dict[T, Any], filter_: Optional[Iterable[T]] = None
) -> dict[T, Any]:
    """
    To filter a dictionary's keys.
    """
    if filter_ is None:
        return dict_
    return {key: value for key, value in dict_.items() if key in filter_}
